"""LoCoMo benchmark evaluation for memory pipeline.

Compares Recall@K across strategies:
  1. Baseline: raw conversation turns ingested directly
  2. Extracted: LLM-extracted facts only (mem0-style pipeline)
  3. Hybrid: raw turns + extracted facts

Usage:
  python scripts/test_locomo.py --local data/locomo.json              # baseline only (fast)
  python scripts/test_locomo.py --local data/locomo.json --limit 1    # 1 sample
  python scripts/test_locomo.py --local data/locomo.json --strategy all  # all 3 strategies (slow)
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import sys
import time
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
from tqdm import tqdm

load_dotenv()

from agent.storage import get_db_url, get_embeddings

BATCH_SIZE = 10  # DashScope embedding batch limit


# ============================================================
# Dataset Loading
# ============================================================

def load_locomo(local_path: str | None = None, limit: int | None = None) -> list[dict]:
    """Load LoCoMo dataset from HuggingFace or local JSON file."""
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
                print("[error] No dataset found. Place locomo.json in backend/data/")
                sys.exit(1)

    if limit:
        samples = samples[:limit]

    total_turns = sum(len(t["dialogue"]) for s in samples for t in s["conversation"])
    total_q = sum(len(s["questions"]) for s in samples)
    print(f"Loaded {len(samples)} samples, {total_turns} turns, {total_q} questions.")
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
    conversation = row.get("conversation") or row.get("dialogue") or row.get("sessions")
    if not conversation:
        return None

    normalized_sessions = []
    if isinstance(conversation, list):
        for session in conversation:
            if isinstance(session, dict):
                sid = session.get("session_id", session.get("id", 0))
                dialogue = session.get("dialogue", session.get("turns", session.get("messages", [])))
                normalized_sessions.append({"session_id": sid, "dialogue": dialogue})
            elif isinstance(session, list):
                normalized_sessions.append({"session_id": len(normalized_sessions) + 1, "dialogue": session})
    elif isinstance(conversation, dict):
        # Handle KimmoZZZ format: {session_1: [...], session_2: [...], ...}
        for key in sorted(conversation.keys()):
            m = re.match(r"session_(\d+)$", key)
            if m and isinstance(conversation[key], list):
                sid = int(m.group(1))
                dt_key = f"{key}_date_time"
                turns = []
                for t in conversation[key]:
                    if isinstance(t, dict):
                        turns.append({
                            "speaker": t.get("speaker", "unknown"),
                            "utterance": t.get("text", t.get("utterance", "")),
                        })
                normalized_sessions.append({"session_id": sid, "datetime": conversation.get(dt_key, ""), "dialogue": turns})

    questions = row.get("questions") or row.get("qa") or row.get("qa_pairs") or row.get("qas") or []
    CAT_MAP = {1: "single_session", 2: "temporal", 3: "multi_session", 4: "adversarial", 5: "event_summary"}
    normalized_questions = []
    for q in questions:
        if isinstance(q, dict):
            answer = q.get("answer", "")
            if answer is None:
                answer = ""
            cat = q.get("category", q.get("question_type", q.get("type", "unknown")))
            if isinstance(cat, int):
                cat = CAT_MAP.get(cat, f"type_{cat}")
            normalized_questions.append({
                "question": q.get("question", q.get("query", "")),
                "answer": str(answer),
                "question_type": cat,
            })

    if not normalized_sessions:
        return None

    return {
        "sample_id": row.get("sample_id", sample_id),
        "conversation": normalized_sessions,
        "questions": normalized_questions,
    }


# ============================================================
# Database Helpers
# ============================================================

def check_db_connection() -> None:
    """Verify PostgreSQL is reachable. Exit with clear message if not."""
    try:
        conn = psycopg.connect(get_db_url(), connect_timeout=5)
        conn.close()
    except Exception as e:
        print(f"\n[error] Cannot connect to PostgreSQL: {e}")
        print("  Start the database first:  docker compose up -d postgres")
        sys.exit(1)


def clean_namespace(namespace: str) -> None:
    """Delete all entries in a namespace."""
    conn = psycopg.connect(get_db_url())
    result = conn.execute("DELETE FROM archival_memory WHERE namespace = %s", (namespace,))
    conn.commit()
    conn.close()
    print(f"  Cleaned '{namespace}': {result.rowcount} entries removed.")


def _extract_all_turns(samples: list[dict]) -> list[dict]:
    """Flatten all turns from all samples into a list of {content, metadata}."""
    entries = []
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                entries.append({"content": content, "metadata": metadata})
    return entries


def ingest_raw_turns(samples: list[dict], namespace: str = "locomo") -> int:
    """Ingest raw conversation turns into archival memory using batch embedding."""
    embeddings = get_embeddings()
    entries = _extract_all_turns(samples)

    if not entries:
        print(f"  No turns to ingest into '{namespace}'.")
        return 0

    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    # Batch embed
    total = 0
    pbar = tqdm(total=len(entries), desc="  Embedding turns", unit="turn")
    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i : i + BATCH_SIZE]
        contents = [e["content"] for e in batch]
        vectors = embeddings.embed_documents(contents)

        for entry, vec in zip(batch, vectors):
            entry_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO archival_memory (id, namespace, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (entry_id, namespace, entry["content"], json.dumps(entry["metadata"]), Vector(vec)),
            )
            total += 1
        pbar.update(len(batch))
    pbar.close()

    conn.commit()
    conn.close()
    print(f"  Ingested {total} raw turns into '{namespace}'.")
    return total


def ingest_session_context(
    samples: list[dict],
    namespace: str = "locomo_context",
    window_size: int = 5,
    stride: int = 2,
) -> int:
    """Ingest overlapping multi-turn windows for richer semantic context.

    Each window contains `window_size` consecutive turns with `stride` step,
    giving the embedding more context than a single turn.
    """
    embeddings = get_embeddings()

    windows = []
    for sample in samples:
        sid = sample["sample_id"]
        for session in sample["conversation"]:
            session_id = session["session_id"]
            turns = [t for t in session["dialogue"] if isinstance(t, dict)]
            for start in range(0, len(turns), stride):
                chunk = turns[start : start + window_size]
                if not chunk:
                    continue
                lines = []
                for t in chunk:
                    speaker = t.get("speaker", t.get("role", "unknown"))
                    utterance = t.get("utterance", t.get("content", t.get("text", "")))
                    if utterance and utterance.strip():
                        lines.append(f"{speaker}: {utterance}")
                if not lines:
                    continue
                content = f"[Session {session_id}] " + "\n".join(lines)
                windows.append({
                    "content": content,
                    "metadata": {
                        "sample_id": sid,
                        "session_id": session_id,
                        "start_turn": start,
                        "window_size": len(chunk),
                        "source": "locomo_context",
                    },
                })

    if not windows:
        print(f"  No windows to ingest into '{namespace}'.")
        return 0

    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    total = 0
    pbar = tqdm(total=len(windows), desc="  Embedding windows", unit="win")
    for i in range(0, len(windows), BATCH_SIZE):
        batch = windows[i : i + BATCH_SIZE]
        contents = [w["content"] for w in batch]
        vectors = embeddings.embed_documents(contents)

        for entry, vec in zip(batch, vectors):
            entry_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO archival_memory (id, namespace, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (entry_id, namespace, entry["content"], json.dumps(entry["metadata"]), Vector(vec)),
            )
            total += 1
        pbar.update(len(batch))
    pbar.close()

    conn.commit()
    conn.close()
    print(f"  Ingested {total} context windows into '{namespace}'.")
    return total


def ingest_extracted_facts(
    samples: list[dict],
    namespace: str = "locomo_extracted",
    window_size: int = 5,
    stride: int = 3,
) -> int:
    """Extract facts from multi-turn windows via LLM and ingest into archival memory.

    Uses overlapping windows instead of individual turns to preserve
    conversational context, temporal relationships, and speaker attribution.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
        temperature=0.0,
    )

    embeddings = get_embeddings()

    extraction_prompt = """Extract facts from this conversation segment. Each fact must be a self-contained statement that preserves WHO, WHAT, WHEN, and WHERE.

Rules:
- Include the speaker's name in each fact (e.g. "Alex went to Paris", NOT "He went to Paris")
- Preserve temporal context (e.g. "last week", "on Tuesday", "in session 3")
- Preserve relationships between people, places, and events
- Each fact should be answerable as a standalone question/answer
- Include specific names, dates, numbers, preferences, decisions
- Do NOT extract general knowledge or opinions
- If no meaningful facts, return an empty list

Conversation (Session {session_id}):
{conversation}

Respond with ONLY a JSON object: {{"facts": ["fact1", "fact2", ...]}}"""

    # Build multi-turn windows
    windows = []
    for sample in samples:
        sid = sample["sample_id"]
        for session in sample["conversation"]:
            session_id = session["session_id"]
            turns = [t for t in session["dialogue"] if isinstance(t, dict)]
            for start in range(0, len(turns), stride):
                chunk = turns[start : start + window_size]
                if not chunk:
                    continue
                lines = []
                for t in chunk:
                    speaker = t.get("speaker", t.get("role", "unknown"))
                    utterance = t.get("utterance", t.get("content", t.get("text", "")))
                    if utterance and utterance.strip():
                        lines.append(f"{speaker}: {utterance}")
                if not lines:
                    continue
                windows.append({
                    "sid": sid,
                    "session_id": session_id,
                    "start_turn": start,
                    "conversation": "\n".join(lines),
                })

    # Extract facts from each window
    all_facts = []
    for w in tqdm(windows, desc="  Extracting facts", unit="win"):
        try:
            prompt = extraction_prompt.format(
                session_id=w["session_id"],
                conversation=w["conversation"],
            )
            response = llm.invoke(prompt)
            parsed = json.loads(_strip_json_fences(response.content))
            facts = parsed.get("facts", [])
        except Exception:
            facts = []

        for fact in facts:
            if not fact or not fact.strip():
                continue
            all_facts.append({
                "content": fact,
                "metadata": {
                    "sample_id": w["sid"],
                    "session_id": w["session_id"],
                    "start_turn": w["start_turn"],
                    "source": "locomo_extracted",
                    "type": "extracted_fact",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            })

    # Batch embed and insert
    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    total = 0
    for i in range(0, len(all_facts), BATCH_SIZE):
        batch = all_facts[i : i + BATCH_SIZE]
        contents = [f["content"] for f in batch]
        vectors = embeddings.embed_documents(contents)

        for fact_entry, vec in zip(batch, vectors):
            entry_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO archival_memory (id, namespace, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (entry_id, namespace, fact_entry["content"], json.dumps(fact_entry["metadata"]), Vector(vec)),
            )
            total += 1

    conn.commit()
    conn.close()
    print(f"  Extracted {total} facts from {len(windows)} windows into '{namespace}'.")
    return total


def _strip_json_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def search_archival(
    query: str, namespace: str, limit: int = 10, alpha: float = 0.7
) -> list[dict]:
    """Hybrid search in archival memory.

    Uses plainto_tsquery with 'english' config (strips stop words, stems)
    and normalizes ts_rank to [0,1].
    """
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


def search_archival_hyde(
    question: str, namespace: str, llm, limit: int = 10, alpha: float = 0.7
) -> list[dict]:
    """HyDE search: generate hypothetical answer, then search with its embedding.

    This bridges the semantic gap between questions (query-style) and
    stored content (answer-style / declarative).
    """
    hyde_prompt = (
        "Answer this question in 1-2 short sentences. "
        "Be specific with names and details. "
        "If you don't know, guess based on the question context.\n\n"
        f"Question: {question}"
    )
    try:
        response = llm.invoke(hyde_prompt)
        hypothetical = response.content.strip()
    except Exception:
        hypothetical = question

    return search_archival(hypothetical, namespace, limit=limit, alpha=alpha)


# ============================================================
# Evaluation Metrics
# ============================================================

def answer_in_results(answer: str, results: list[dict], k: int) -> bool:
    """Check if answer appears in top-K results (substring + token overlap)."""
    answer_lower = answer.lower().strip()
    if not answer_lower:
        return False

    for r in results[:k]:
        content_lower = r["content"].lower()

        # Substring match
        if answer_lower in content_lower:
            return True

        # Token overlap
        answer_tokens = set(re.findall(r"\w+", answer_lower))
        if len(answer_tokens) >= 2:
            content_tokens = set(re.findall(r"\w+", content_lower))
            overlap = len(answer_tokens & content_tokens)
            if overlap / len(answer_tokens) > 0.5:
                return True

    return False


def llm_judge_answer(
    question: str, answer: str, results: list[dict], k: int, llm
) -> bool:
    """Use LLM to check if any top-K result entails the expected answer."""
    if not answer.strip():
        return False

    judge_prompt = """You are evaluating whether a retrieved passage contains the answer to a question.

Question: {question}
Expected answer: {answer}

Retrieved passages (top {k}):
{passages}

Does ANY of the retrieved passages contain information that answers the question with the expected answer?
Consider paraphrases, synonyms, and implied information. The answer does not need to be word-for-word.

Respond with ONLY "yes" or "no"."""

    passages = []
    for i, r in enumerate(results[:k]):
        passages.append(f"[{i+1}] {r['content'][:500]}")

    try:
        prompt = judge_prompt.format(
            question=question,
            answer=answer,
            k=k,
            passages="\n".join(passages),
        )
        response = llm.invoke(prompt)
        return response.content.strip().lower().startswith("yes")
    except Exception:
        return False


# Type for answer checking functions: (question, answer, results, k) -> bool
AnswerChecker = callable


def _default_checker(question: str, answer: str, results: list[dict], k: int) -> bool:
    """Default answer checker: substring + token overlap. Ignores question."""
    return answer_in_results(answer, results, k)


def evaluate_recall(
    samples: list[dict],
    namespace: str,
    k_values: list[int] | None = None,
    answer_checker=None,
    search_fn=None,
) -> dict:
    """Evaluate Recall@K for all questions.

    Args:
        answer_checker: optional callable(question, answer, results, k) -> bool.
        search_fn: optional callable(question, namespace, limit) -> list[dict].
            Defaults to search_archival. Use functools.partial to bind an LLM for HyDE.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    max_k = max(k_values)
    results: dict[str, dict[int, dict]] = {}

    all_questions = []
    for sample in samples:
        for q in sample["questions"]:
            all_questions.append(q)

    checker = answer_checker or _default_checker
    searcher = search_fn or search_archival

    t0 = time.time()

    for q in tqdm(all_questions, desc="  Evaluating R@K", unit="q"):
        qtype = q.get("question_type", "unknown")
        question = q.get("question", "")
        answer = q.get("answer", "")

        if not question or not answer:
            continue

        if qtype not in results:
            results[qtype] = {k: {"hits": 0, "total": 0} for k in k_values}

        for k in k_values:
            results[qtype][k]["total"] += 1

        search_results = searcher(question, namespace, limit=max_k)

        for k in k_values:
            if checker(question, answer, search_results, k):
                results[qtype][k]["hits"] += 1

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
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
    """Run evaluation under 4 strategies and compare."""
    k_values = [1, 3, 5, 10]

    print("\n" + "=" * 70)
    print("  LoCoMo Memory Pipeline Evaluation")
    print("=" * 70)

    # Strategy 1: Baseline
    print("\n--- Strategy 1: Baseline (raw conversation turns) ---")
    ns = "locomo_baseline"
    clean_namespace(ns)
    ingest_raw_turns(samples, namespace=ns)
    results_baseline = evaluate_recall(samples, ns, k_values)
    print_results(results_baseline, "Baseline: Raw Turns")

    # Strategy 2: Session context windows
    print("\n--- Strategy 2: Session context windows ---")
    ns = "locomo_context"
    clean_namespace(ns)
    ingest_session_context(samples, namespace=ns)
    results_context = evaluate_recall(samples, ns, k_values)
    print_results(results_context, "Session Context Windows")

    # Strategy 3: Extracted facts
    print("\n--- Strategy 3: Extracted facts (mem0 pipeline) ---")
    ns = "locomo_extracted"
    clean_namespace(ns)
    ingest_extracted_facts(samples, namespace=ns)
    results_extracted = evaluate_recall(samples, ns, k_values)
    print_results(results_extracted, "Extracted Facts")

    # Strategy 4: Hybrid (context + extracted)
    print("\n--- Strategy 4: Hybrid (context + extracted) ---")
    ns = "locomo_hybrid"
    clean_namespace(ns)
    ingest_session_context(samples, namespace=ns)
    ingest_extracted_facts(samples, namespace=ns)
    results_hybrid = evaluate_recall(samples, ns, k_values)
    print_results(results_hybrid, "Hybrid: Context + Extracted")

    # Summary
    print("\n" + "=" * 70)
    print("  Summary: Overall Recall@K Comparison")
    print("=" * 70)
    print(f"  {'Strategy':<25}", end="")
    for k in k_values:
        print(f"{'R@' + str(k):>8}", end="")
    print()
    print("  " + "-" * (25 + 8 * len(k_values)))

    strategies = [
        ("Baseline (raw)", results_baseline),
        ("Context windows", results_context),
        ("Extracted (mem0)", results_extracted),
        ("Hybrid", results_hybrid),
    ]
    for label, res in strategies:
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
    from functools import partial

    parser = argparse.ArgumentParser(description="LoCoMo memory pipeline evaluation")
    parser.add_argument("--local", type=str, default=None, help="Path to local LoCoMo JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument(
        "--strategy", type=str, default="baseline",
        choices=["all", "baseline", "context", "extracted", "hybrid"],
        help="Which strategy to evaluate (default: baseline)",
    )
    parser.add_argument("--hyde", action="store_true", help="Use HyDE (hypothetical answer) for query rewriting")
    parser.add_argument("--llm-judge", action="store_true", help="Use LLM for answer matching instead of substring/token")
    args = parser.parse_args()

    # Check DB first
    check_db_connection()

    t_start = time.time()
    samples = load_locomo(local_path=args.local, limit=args.limit)

    # Build LLM if needed (shared between HyDE and judge)
    llm = None
    if args.hyde or args.llm_judge:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("LLM_API_KEY", ""),
            temperature=0.0,
        )

    # Build search function
    search_fn = None
    if args.hyde:
        search_fn = partial(search_archival_hyde, llm=llm)
        print("  Using HyDE query rewriting.")

    # Build answer checker
    checker = None
    if args.llm_judge:
        def checker(question, answer, results, k, _llm=llm):
            return llm_judge_answer(question, answer, results, k, _llm)
        print("  Using LLM judge for answer matching.")

    if args.strategy == "all":
        compare_strategies(samples)
    elif args.strategy == "baseline":
        ns = "locomo_baseline"
        clean_namespace(ns)
        ingest_raw_turns(samples, namespace=ns)
        results = evaluate_recall(samples, ns, answer_checker=checker, search_fn=search_fn)
        print_results(results, "Baseline: Raw Turns")
    elif args.strategy == "context":
        ns = "locomo_context"
        clean_namespace(ns)
        ingest_session_context(samples, namespace=ns)
        results = evaluate_recall(samples, ns, answer_checker=checker, search_fn=search_fn)
        print_results(results, "Session Context Windows")
    elif args.strategy == "extracted":
        ns = "locomo_extracted"
        clean_namespace(ns)
        ingest_extracted_facts(samples, namespace=ns)
        results = evaluate_recall(samples, ns, answer_checker=checker, search_fn=search_fn)
        print_results(results, "Extracted Facts")
    elif args.strategy == "hybrid":
        ns = "locomo_hybrid"
        clean_namespace(ns)
        ingest_session_context(samples, namespace=ns)
        ingest_extracted_facts(samples, namespace=ns)
        results = evaluate_recall(samples, ns, answer_checker=checker, search_fn=search_fn)
        print_results(results, "Hybrid: Context + Extracted")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")
