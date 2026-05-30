"""Focused evaluation for specific LoCoMo question types.

Test only event_summary, temporal, or other question types in isolation.

Usage:
  python scripts/test_question_type.py --type event_summary
  python scripts/test_question_type.py --type temporal --details
  python scripts/test_question_type.py --type event_summary,temporal --optimized
  python scripts/test_question_type.py --type event_summary --compare-judge --details
  python scripts/test_question_type.py --type temporal --compare-strategies
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
import traceback
import warnings

# Suppress warnings before any imports
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except ImportError:
    pass

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("[1/3] Importing test_locomo...", flush=True)
from test_locomo import (
    build_evidence_index,
    check_db_connection,
    clean_namespace,
    evaluate_recall,
    ingest_session_context,
    ingest_extracted_facts,
    ingest_cross_session_summary,
    load_locomo,
    llm_judge_answer,
    search_archival,
    print_results,
    answer_in_results_with_evidence,
)
print("[2/3] test_locomo imported.", flush=True)


def get_optimized_search():
    """Lazy import of optimized search."""
    from run_optimized_benchmark import search_archival_optimized
    return search_archival_optimized


def filter_questions_by_type(samples, types):
    """Filter samples to only include questions of specified types."""
    filtered = []
    for sample in samples:
        filtered_questions = [
            q for q in sample["questions"]
            if q.get("question_type") in types
        ]
        if filtered_questions:
            filtered.append({**sample, "questions": filtered_questions})
    return filtered


def evaluate_with_details(
    samples, namespace, k_values,
    use_llm_judge=False, use_optimized=False, llm=None,
):
    """Evaluate and return both results and per-question details."""
    max_k = max(k_values)
    results = {}
    details = []
    evidence_idx = build_evidence_index(samples)

    all_questions = []
    for sample in samples:
        for q in sample["questions"]:
            all_questions.append(q)

    optimized_search = get_optimized_search() if use_optimized else None

    for i, q in enumerate(all_questions):
        qtype = q.get("question_type", "unknown")
        question = q.get("question", "")
        answer = q.get("answer", "")
        evidence = q.get("evidence", [])

        if not question or (not answer.strip() and not evidence):
            continue

        if qtype not in results:
            results[qtype] = {k: {"hits": 0, "total": 0} for k in k_values}

        for k in k_values:
            results[qtype][k]["total"] += 1

        # Search
        print(f"  [{i+1}/{len(all_questions)}] {question[:60]}...", flush=True)
        if use_optimized and optimized_search:
            search_results = optimized_search(
                question, namespace, limit=max_k,
                enable_hyde=True, enable_dedup=True, enable_mmr=True, llm=llm,
            )
        else:
            search_results = search_archival(question, namespace, limit=max_k)

        # Checker
        def checker(chk_q, chk_a, chk_res, chk_k, evidence=None):
            if answer_in_results_with_evidence(chk_a, chk_res, chk_k, evidence, evidence_idx):
                return True
            if use_llm_judge and llm and chk_a.strip():
                return llm_judge_answer(chk_q, chk_a, chk_res, chk_k, llm)
            return False

        # Recall@K
        hit_at = []
        for k in k_values:
            if checker(question, answer, search_results, k, evidence=evidence):
                results[qtype][k]["hits"] += 1
                hit_at.append(k)

        # MRR
        first_rank = 0
        for rank_idx, r in enumerate(search_results[:max_k]):
            if checker(question, answer, [r], 1, evidence=evidence):
                first_rank = rank_idx + 1
                break
        rr = 1.0 / first_rank if first_rank > 0 else 0.0

        if "_mrr" not in results:
            results["_mrr"] = {}
        if qtype not in results["_mrr"]:
            results["_mrr"][qtype] = {"rr_sum": 0.0, "total": 0}
        results["_mrr"][qtype]["rr_sum"] += rr
        results["_mrr"][qtype]["total"] += 1

        detail = {
            "question": question,
            "answer": answer,
            "type": qtype,
            "hit_at": hit_at,
            "first_rank": first_rank,
            "rr": rr,
        }
        if search_results:
            detail["top1"] = search_results[0]["content"][:200]
            detail["top1_score"] = search_results[0].get("score", 0)
            detail["top3_contents"] = [r["content"][:150] for r in search_results[:3]]
        details.append(detail)

    return results, details


def print_type_summary(results, title, k_values):
    """Print compact summary."""
    print(f"\n  {title}", flush=True)
    header = f"  {'Type':<20}"
    for k in k_values:
        header += f"{'R@' + str(k):>8}"
    header += f"{'MRR':>8}"
    print(header, flush=True)
    print("  " + "-" * (20 + 8 * len(k_values) + 8), flush=True)

    for qtype in sorted(k for k in results.keys() if k != "_mrr"):
        line = f"  {qtype:<20}"
        for k in k_values:
            data = results[qtype].get(k, {"hits": 0, "total": 0})
            recall = data["hits"] / data["total"] if data["total"] > 0 else 0
            line += f"{recall:>7.1%} "
        if "_mrr" in results and qtype in results["_mrr"]:
            mrr_data = results["_mrr"][qtype]
            mrr = mrr_data["rr_sum"] / mrr_data["total"] if mrr_data["total"] > 0 else 0
            line += f"{mrr:>7.1%} "
        print(line, flush=True)


def print_details(details, k_values):
    """Print per-question details."""
    misses = [d for d in details if not d["hit_at"]]
    hits = [d for d in details if d["hit_at"]]

    print(f"\n  Hits: {len(hits)}, Misses: {len(misses)}", flush=True)

    if misses:
        print(f"\n  --- MISSED QUESTIONS (not in top-{max(k_values)}) ---", flush=True)
        for d in misses:
            print(f"\n  Q: {d['question']}", flush=True)
            print(f"  A: {d['answer']}", flush=True)
            print(f"  Top1 score: {d.get('top1_score', 0):.3f}", flush=True)
            if d.get("top3_contents"):
                for i, c in enumerate(d["top3_contents"]):
                    print(f"  [{i+1}] {c}...", flush=True)

    if hits:
        print(f"\n  --- HIT QUESTIONS ---", flush=True)
        for d in hits:
            print(f"\n  Q: {d['question']}", flush=True)
            print(f"  A: {d['answer']}", flush=True)
            print(f"  Hit at K={d['hit_at']}, first_rank={d['first_rank']}, RR={d['rr']:.3f}", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Focused evaluation for specific LoCoMo question types"
    )
    parser.add_argument(
        "--type", type=str, required=True,
        help="Question type(s), comma-separated: event_summary,temporal,single_session,multi_session,adversarial"
    )
    parser.add_argument("--local", type=str, default=None, help="Path to local LoCoMo JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples")
    parser.add_argument("--details", action="store_true", help="Show per-question details")
    parser.add_argument("--optimized", action="store_true", help="Use optimized search (HyDE+MMR+dedup)")
    parser.add_argument("--compare-judge", action="store_true", help="Compare with/without LLM judge")
    parser.add_argument("--compare-strategies", action="store_true", help="Compare raw/context/hybrid")
    parser.add_argument("--output", type=str, default=None, help="Export results to JSON")
    args = parser.parse_args()

    check_db_connection()

    target_types = [t.strip() for t in args.type.split(",")]
    print(f"  Question types: {target_types}", flush=True)

    samples = load_locomo(local_path=args.local, limit=args.limit)
    filtered = filter_questions_by_type(samples, target_types)

    total_questions = sum(len(s["questions"]) for s in filtered)
    print(f"  Found {total_questions} questions across {len(filtered)} samples", flush=True)

    if total_questions == 0:
        print("  No questions found. Exiting.", flush=True)
        return

    k_values = [1, 3, 5, 10]

    # Initialize LLM if needed
    llm = None
    if args.optimized or args.compare_judge:
        try:
            from langchain_openai import ChatOpenAI
            from agent.configuration import Configuration
            config = Configuration.from_runnable_config()
            llm = ChatOpenAI(
                model=config.llm_model,
                base_url=config.llm_base_url,
                api_key=config.llm_api_key,
                temperature=0,
            )
            print(f"  LLM: {config.llm_model}", flush=True)
        except Exception as e:
            print(f"  [warn] LLM init failed: {e}", flush=True)

    print("[3/3] Starting evaluation...", flush=True)

    if args.compare_strategies:
        strategies = [
            ("Baseline (raw turns)", "locomo_type_baseline", False, False),
            ("Context windows", "locomo_type_context", True, False),
            ("Hybrid (ctx+facts+cross)", "locomo_type_hybrid", True, True),
        ]
        all_results = {}

        for label, ns, use_context, use_hybrid in strategies:
            print(f"\n  --- {label} ---", flush=True)
            clean_namespace(ns)
            if use_context:
                ingest_session_context(filtered, namespace=ns)
            else:
                from test_locomo import ingest_raw_turns
                ingest_raw_turns(filtered, namespace=ns)
            if use_hybrid:
                ingest_extracted_facts(filtered, namespace=ns)
                ingest_cross_session_summary(filtered, namespace=ns)

            results, details = evaluate_with_details(
                filtered, ns, k_values,
                use_llm_judge=llm is not None,
                use_optimized=args.optimized, llm=llm,
            )
            all_results[label] = results
            print_type_summary(results, label, k_values)
            if args.details:
                print_details(details, k_values)
            clean_namespace(ns)

        print("\n  " + "=" * 60, flush=True)
        print("  Strategy Comparison", flush=True)
        print("  " + "=" * 60, flush=True)
        for label, results in all_results.items():
            for qtype in sorted(k for k in results.keys() if k != "_mrr"):
                d5 = results[qtype].get(5, {"hits": 0, "total": 0})
                r5 = d5["hits"] / d5["total"] if d5["total"] > 0 else 0
                print(f"  {label:<30} {qtype:<15} R@5={r5:.1%}", flush=True)

    elif args.compare_judge:
        ns = "locomo_type_judge"
        clean_namespace(ns)
        ingest_session_context(filtered, namespace=ns)
        ingest_extracted_facts(filtered, namespace=ns)
        ingest_cross_session_summary(filtered, namespace=ns)

        print("\n  --- Without LLM Judge ---", flush=True)
        res_no, det_no = evaluate_with_details(
            filtered, ns, k_values, use_llm_judge=False,
            use_optimized=args.optimized, llm=llm,
        )
        print_type_summary(res_no, "Without LLM Judge", k_values)

        print("\n  --- With LLM Judge ---", flush=True)
        res_yes, det_yes = evaluate_with_details(
            filtered, ns, k_values, use_llm_judge=True,
            use_optimized=args.optimized, llm=llm,
        )
        print_type_summary(res_yes, "With LLM Judge", k_values)

        if args.details:
            print("\n  --- Judge Decision Differences ---", flush=True)
            for d_no, d_yes in zip(det_no, det_yes):
                if d_no["hit_at"] != d_yes["hit_at"]:
                    print(f"\n  Q: {d_no['question']}", flush=True)
                    print(f"  A: {d_no['answer']}", flush=True)
                    print(f"  Without judge: hit_at={d_no['hit_at']}", flush=True)
                    print(f"  With judge:    hit_at={d_yes['hit_at']}", flush=True)
        clean_namespace(ns)

    else:
        ns = "locomo_type_single"
        clean_namespace(ns)

        print("\n  Ingesting...", flush=True)
        ingest_session_context(filtered, namespace=ns)
        if llm:
            ingest_extracted_facts(filtered, namespace=ns)
            ingest_cross_session_summary(filtered, namespace=ns)

        print("\n  Evaluating...", flush=True)
        results, details = evaluate_with_details(
            filtered, ns, k_values,
            use_llm_judge=llm is not None,
            use_optimized=args.optimized, llm=llm,
        )

        print_type_summary(results, f"Results ({', '.join(target_types)})", k_values)

        if args.details:
            print_details(details, k_values)

        clean_namespace(ns)

        if args.output:
            export = {
                "question_types": target_types,
                "total_questions": total_questions,
                "results": results,
                "details": details if args.details else None,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(export, f, indent=2, ensure_ascii=False)
            print(f"\n  Exported to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
