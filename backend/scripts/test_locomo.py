"""LoCoMo benchmark evaluation for memory pipeline.

Compares Recall@K across three strategies:
  1. Baseline: raw conversation turns ingested directly
  2. Extracted: LLM-extracted facts only (mem0-style pipeline)
  3. Hybrid: raw turns + extracted facts

Usage:
  python scripts/test_locomo.py                     # HuggingFace dataset
  python scripts/test_locomo.py --local data/locomo.json  # local JSON file
  python scripts/test_locomo.py --limit 5           # only first N samples
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

load_dotenv()

from agent.storage import get_db_url, get_embeddings


# ============================================================
# Dataset Loading
# ============================================================

def load_locomo(local_path: str | None = None, limit: int | None = None) -> list[dict]:
    """Load LoCoMo dataset from HuggingFace or local JSON file.

    Returns list of dicts, each with:
      - sample_id: str
      - conversation: list of sessions, each with dialogue turns
      - questions: list of {question, answer, question_type}
    """
    samples = []

    if local_path:
        samples = _load_local(local_path)
    else:
        try:
            samples = _load_huggingface()
        except Exception as e:
            print(f"[warn] HuggingFace load failed: {e}")
            fallback = os.path.join(os.path.dirname(__file__), "..", "data", "locomo.json")
            if os.path.exists(fallback):
                print(f"[info] Falling back to local file: {fallback}")
                samples = _load_local(fallback)
            else:
                print("[error] No dataset found. Place locomo.json in backend/data/ or install `datasets`.")
                sys.exit(1)

    if limit:
        samples = samples[:limit]

    print(f"Loaded {len(samples)} LoCoMo samples.")
    return samples


def _load_huggingface() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("locomo/locomo", split="test")
    samples = []
    for i, row in enumerate(ds):
        sample = _normalize_sample(row, sample_id=str(i))
        if sample:
            samples.append(sample)
    return samples


def _load_local(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        raw_samples = data
    elif isinstance(data, dict):
        # Handle HuggingFace-style {"train": [...], "test": [...]} or single sample
        if "test" in data:
            raw_samples = data["test"]
        elif "train" in data:
            raw_samples = data["train"]
        else:
            raw_samples = [data]
    else:
        raw_samples = []

    samples = []
    for i, row in enumerate(raw_samples):
        sample = _normalize_sample(row, sample_id=str(i))
        if sample:
            samples.append(sample)
    return samples


def _normalize_sample(row: dict, sample_id: str) -> dict | None:
    """Normalize different LoCoMo format variants into a common structure."""
    # Extract conversation
    conversation = row.get("conversation") or row.get("dialogue") or row.get("sessions")
    if not conversation:
        return None

    # Normalize conversation to list of sessions with dialogue
    normalized_sessions = []
    if isinstance(conversation, list):
        for session in conversation:
            if isinstance(session, dict):
                sid = session.get("session_id", session.get("id", 0))
                dialogue = session.get("dialogue", session.get("turns", session.get("messages", [])))
                normalized_sessions.append({"session_id": sid, "dialogue": dialogue})
            elif isinstance(session, list):
                normalized_sessions.append({"session_id": len(normalized_sessions) + 1, "dialogue": session})

    # Extract questions
    questions = row.get("questions") or row.get("qa_pairs") or row.get("qas") or []
    normalized_questions = []
    for q in questions:
        if isinstance(q, dict):
            normalized_questions.append({
                "question": q.get("question", q.get("query", "")),
                "answer": q.get("answer", q.get("response", "")),
                "question_type": q.get("question_type", q.get("type", "unknown")),
            })

    if not normalized_sessions:
        return None

    return {
        "sample_id": sample_id,
        "conversation": normalized_sessions,
        "questions": normalized_questions,
    }


# ============================================================
# Database Helpers
# ============================================================

def clean_namespace(namespace: str) -> None:
    """Delete all entries in a namespace."""
    conn = psycopg.connect(get_db_url())
    result = conn.execute("DELETE FROM archival_memory WHERE namespace = %s", (namespace,))
    conn.commit()
    conn.close()
    print(f"  Cleaned namespace '{namespace}': {result.rowcount} entries removed.")


def ingest_raw_turns(samples: list[dict], namespace: str = "locomo") -> int:
    """Ingest raw conversation turns into archival memory."""
    embeddings = get_embeddings()
    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    total = 0
    for sample in samples:
        sid = sample["sample_id"]
        for session in sample["conversation"]:
            session_id = session["session_id"]
            for turn_idx, turn in enumerate(session["dialogue"]):
                if not isinstance(turn, dict):
                    continue
                speaker = turn.get("speaker", turn.get("role", "unknown"))
                utterance = turn.get("utterance", turn.get("content", turn.get("text", "")))
                if not utterance or not utterance.strip():
                    continue

                content = f"[{speaker}] {utterance}"
                metadata = {
                    "sample_id": sid,
                    "session_id": session_id,
                    "turn_idx": turn_idx,
                    "speaker": speaker,
                    "source": "locomo_raw",
                }
                entry_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                metadata["timestamp"] = now

                # Batch embedding would be more efficient but we need per-entry metadata
                vec = embeddings.embed_query(content)
                conn.execute(
                    "INSERT INTO archival_memory (id, namespace, content, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (entry_id, namespace, content, json.dumps(metadata), Vector(vec)),
                )
                total += 1

        # Commit per sample to avoid huge transactions
        conn.commit()

    conn.close()
    print(f"  Ingested {total} raw turns into '{namespace}'.")
    return total


def ingest_extracted_facts(samples: list[dict], namespace: str = "locomo_extracted") -> int:
    """Extract facts from conversations via LLM and ingest into archival memory."""
    from langchain_openai import ChatOpenAI
    from agent.configuration import Configuration

    config = Configuration()
    llm = ChatOpenAI(
        model=config.llm_model,
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        temperature=0.0,
        extra_body={"thinking": {"type": "enabled"}},
    )

    embeddings = get_embeddings()
    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    extraction_prompt = """Extract discrete facts from this conversation turn. Each fact must be a single, self-contained statement.

Rules:
- Each fact is one sentence, standalone (no pronouns without referent)
- Include specific names, dates, numbers, preferences, decisions, relationships
- Do NOT extract general knowledge or opinions
- If no meaningful facts, return an empty list

Speaker: {speaker}
Message: {message}

Respond with ONLY a JSON object: {{"facts": ["fact1", ...]}}"""

    total = 0
    for sample in samples:
        sid = sample["sample_id"]
        for session in sample["conversation"]:
            session_id = session["session_id"]
            for turn_idx, turn in enumerate(session["dialogue"]):
                if not isinstance(turn, dict):
                    continue
                speaker = turn.get("speaker", turn.get("role", "unknown"))
                utterance = turn.get("utterance", turn.get("content", turn.get("text", "")))
                if not utterance or not utterance.strip():
                    continue

                # Extract facts via LLM
                try:
                    prompt = extraction_prompt.format(speaker=speaker, message=utterance)
                    response = llm.invoke(prompt)
                    parsed = json.loads(_strip_json_fences(response.content))
                    facts = parsed.get("facts", [])
                except Exception:
                    facts = []

                for fact in facts:
                    if not fact or not fact.strip():
                        continue
                    metadata = {
                        "sample_id": sid,
                        "session_id": session_id,
                        "source": "locomo_extracted",
                        "type": "extracted_fact",
                    }
                    entry_id = str(uuid.uuid4())
                    metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

                    vec = embeddings.embed_query(fact)
                    conn.execute(
                        "INSERT INTO archival_memory (id, namespace, content, metadata, embedding) "
                        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                        (entry_id, namespace, fact, json.dumps(metadata), Vector(vec)),
                    )
                    total += 1

        conn.commit()

    conn.close()
    print(f"  Ingested {total} extracted facts into '{namespace}'.")
    return total


def _strip_json_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def search_archival(
    query: str, namespace: str, limit: int = 10, alpha: float = 0.7
) -> list[dict]:
    """Hybrid search in archival memory (same pattern as test_recall.py)."""
    embeddings = get_embeddings()
    query_embedding = Vector(embeddings.embed_query(query))
    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    rows = conn.execute(
        """
        SELECT content, metadata,
               %s * (1 - (embedding <=> %s::vector))
                 + (1 - %s) * ts_rank(content_tsv, plainto_tsquery('simple', %s))
               AS score
        FROM archival_memory
        WHERE namespace = %s
          AND (content_tsv @@ plainto_tsquery('simple', %s)
               OR 1 - (embedding <=> %s::vector) > 0.2)
        ORDER BY score DESC
        LIMIT %s
        """,
        (alpha, query_embedding, alpha, query, namespace, query, query_embedding, limit),
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
# Evaluation Metrics
# ============================================================

def answer_in_results(answer: str, results: list[dict], k: int) -> bool:
    """Check if answer appears in top-K results (substring + token overlap)."""
    answer_lower = answer.lower().strip()
    if not answer_lower:
        return False

    top_k = results[:k]
    for r in top_k:
        content_lower = r["content"].lower()

        # Substring match
        if answer_lower in content_lower:
            return True

        # Token overlap: tokenize answer, check if >50% tokens appear in content
        answer_tokens = set(re.findall(r"\w+", answer_lower))
        if len(answer_tokens) >= 2:
            content_tokens = set(re.findall(r"\w+", content_lower))
            overlap = len(answer_tokens & content_tokens)
            if overlap / len(answer_tokens) > 0.5:
                return True

    return False


def evaluate_recall(
    samples: list[dict],
    namespace: str,
    k_values: list[int] | None = None,
) -> dict:
    """Evaluate Recall@K for all questions in the dataset.

    Returns: {question_type: {k: {"hits": int, "total": int}}}
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    max_k = max(k_values)
    results: dict[str, dict[int, dict]] = {}
    all_questions = []

    for sample in samples:
        sample_id = sample["sample_id"]
        for q in sample["questions"]:
            q["sample_id"] = sample_id
            all_questions.append(q)

    print(f"  Evaluating {len(all_questions)} questions (namespace='{namespace}')...")

    for q in all_questions:
        qtype = q.get("question_type", "unknown")
        question = q.get("question", "")
        answer = q.get("answer", "")

        if not question or not answer:
            continue

        if qtype not in results:
            results[qtype] = {k: {"hits": 0, "total": 0} for k in k_values}

        for k in k_values:
            results[qtype][k]["total"] += 1

        # Search once with max_k
        search_results = search_archival(question, namespace, limit=max_k)

        for k in k_values:
            if answer_in_results(answer, search_results, k):
                results[qtype][k]["hits"] += 1

    return results


def print_results(results: dict, title: str) -> None:
    """Print recall results as a formatted table."""
    k_values = set()
    for qtype_data in results.values():
        k_values.update(qtype_data.keys())
    k_values = sorted(k_values)

    print(f"\n  {title}")
    print(f"  {'Question Type':<20}", end="")
    for k in k_values:
        print(f"{'R@' + str(k):>8}", end="")
    print()
    print("  " + "-" * (20 + 8 * len(k_values)))

    total_hits = {k: 0 for k in k_values}
    total_count = {k: 0 for k in k_values}

    for qtype in sorted(results.keys()):
        print(f"  {qtype:<20}", end="")
        for k in k_values:
            data = results[qtype].get(k, {"hits": 0, "total": 1})
            recall = data["hits"] / data["total"] if data["total"] > 0 else 0
            print(f"{recall:>7.1%}", end=" ")
            total_hits[k] += data["hits"]
            total_count[k] += data["total"]
        print()

    print("  " + "-" * (20 + 8 * len(k_values)))
    print(f"  {'Overall':<20}", end="")
    for k in k_values:
        recall = total_hits[k] / total_count[k] if total_count[k] > 0 else 0
        print(f"{recall:>7.1%}", end=" ")
    print()


# ============================================================
# Strategy Comparison
# ============================================================

def compare_strategies(samples: list[dict]) -> None:
    """Run evaluation under 3 strategies and compare."""
    k_values = [1, 3, 5, 10]

    print("\n" + "=" * 70)
    print("  LoCoMo Memory Pipeline Evaluation")
    print("=" * 70)

    # --- Strategy 1: Baseline (raw turns) ---
    print("\n--- Strategy 1: Baseline (raw conversation turns) ---")
    ns_baseline = "locomo_baseline"
    clean_namespace(ns_baseline)
    ingest_raw_turns(samples, namespace=ns_baseline)
    results_baseline = evaluate_recall(samples, ns_baseline, k_values)
    print_results(results_baseline, "Baseline: Raw Turns")

    # --- Strategy 2: Extracted facts only ---
    print("\n--- Strategy 2: Extracted facts (mem0 pipeline) ---")
    ns_extracted = "locomo_extracted"
    clean_namespace(ns_extracted)
    ingest_extracted_facts(samples, namespace=ns_extracted)
    results_extracted = evaluate_recall(samples, ns_extracted, k_values)
    print_results(results_extracted, "Extracted Facts")

    # --- Strategy 3: Hybrid (raw + extracted) ---
    print("\n--- Strategy 3: Hybrid (raw turns + extracted facts) ---")
    ns_hybrid = "locomo_hybrid"
    clean_namespace(ns_hybrid)
    ingest_raw_turns(samples, namespace=ns_hybrid)
    ingest_extracted_facts(samples, namespace=ns_hybrid)
    results_hybrid = evaluate_recall(samples, ns_hybrid, k_values)
    print_results(results_hybrid, "Hybrid: Raw + Extracted")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  Summary: Overall Recall@K Comparison")
    print("=" * 70)
    print(f"  {'Strategy':<25}", end="")
    for k in k_values:
        print(f"{'R@' + str(k):>8}", end="")
    print()
    print("  " + "-" * (25 + 8 * len(k_values)))

    for label, res in [("Baseline (raw)", results_baseline), ("Extracted (mem0)", results_extracted), ("Hybrid", results_hybrid)]:
        total_hits = {k: 0 for k in k_values}
        total_count = {k: 0 for k in k_values}
        for qtype_data in res.values():
            for k in k_values:
                data = qtype_data.get(k, {"hits": 0, "total": 0})
                total_hits[k] += data["hits"]
                total_count[k] += data["total"]
        print(f"  {label:<25}", end="")
        for k in k_values:
            recall = total_hits[k] / total_count[k] if total_count[k] > 0 else 0
            print(f"{recall:>7.1%}", end=" ")
        print()

    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoCoMo memory pipeline evaluation")
    parser.add_argument("--local", type=str, default=None, help="Path to local LoCoMo JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["all", "baseline", "extracted", "hybrid"],
                        help="Which strategy to evaluate")
    args = parser.parse_args()

    samples = load_locomo(local_path=args.local, limit=args.limit)

    if args.strategy == "all":
        compare_strategies(samples)
    elif args.strategy == "baseline":
        ns = "locomo_baseline"
        clean_namespace(ns)
        ingest_raw_turns(samples, namespace=ns)
        results = evaluate_recall(samples, ns)
        print_results(results, "Baseline: Raw Turns")
    elif args.strategy == "extracted":
        ns = "locomo_extracted"
        clean_namespace(ns)
        ingest_extracted_facts(samples, namespace=ns)
        results = evaluate_recall(samples, ns)
        print_results(results, "Extracted Facts")
    elif args.strategy == "hybrid":
        ns = "locomo_hybrid"
        clean_namespace(ns)
        ingest_raw_turns(samples, namespace=ns)
        ingest_extracted_facts(samples, namespace=ns)
        results = evaluate_recall(samples, ns)
        print_results(results, "Hybrid: Raw + Extracted")
