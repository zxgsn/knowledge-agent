"""Quick LoCoMo evaluation: extracted facts with large stride (fewer LLM calls)."""
from __future__ import annotations
import asyncio, io, json, os, re, sys, time, uuid
from datetime import datetime, timezone

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector
from tqdm import tqdm

load_dotenv()
from agent.storage import get_db_url, get_embeddings
from langchain_openai import ChatOpenAI

BATCH_SIZE = 10
CAT_MAP = {1: "single_session", 2: "temporal", 3: "multi_session", 4: "adversarial", 5: "event_summary"}


def load_samples(path: str, limit: int = 1) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw_list = raw if isinstance(raw, list) else raw.get("test", raw.get("train", [raw]))
    samples = []
    for i, row in enumerate(raw_list[:limit]):
        conv = row.get("conversation", row.get("dialogue", row.get("sessions", [])))
        sessions = []
        if isinstance(conv, dict):
            for key in sorted(conv.keys()):
                m = re.match(r"session_(\d+)$", key)
                if m and isinstance(conv[key], list):
                    turns = [{"speaker": t.get("speaker", "?"), "utterance": t.get("text", t.get("utterance", ""))} for t in conv[key] if isinstance(t, dict)]
                    sessions.append({"session_id": int(m.group(1)), "dialogue": turns})
        elif isinstance(conv, list):
            for sess in conv:
                if isinstance(sess, dict):
                    sid = sess.get("session_id", sess.get("id", len(sessions) + 1))
                    turns = sess.get("dialogue", sess.get("turns", []))
                    sessions.append({"session_id": sid, "dialogue": turns})
        qs = []
        for q in row.get("questions", []):
            ans = q.get("answer", "")
            cat = q.get("category", q.get("question_type", "unknown"))
            if isinstance(cat, int):
                cat = CAT_MAP.get(cat, f"type_{cat}")
            qs.append({"question": q.get("question", ""), "answer": str(ans or ""), "question_type": cat})
        samples.append({"sample_id": str(i), "conversation": sessions, "questions": qs})
    return samples


def extract_facts(samples: list[dict], window_size: int = 10, stride: int = 20) -> list[dict]:
    """Extract facts from large windows (fewer LLM calls)."""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=0.0,
    )

    prompt_tpl = (
        "Extract facts from this conversation segment. Each fact must be self-contained, "
        "preserving WHO, WHAT, WHEN, WHERE.\n"
        "Rules:\n"
        "- Include speaker name in each fact\n"
        "- Preserve temporal context\n"
        "- Preserve relationships between people, places, events\n"
        "- Include specific names, dates, numbers, preferences\n"
        "- Do NOT extract general knowledge or opinions\n"
        "- If no meaningful facts, return empty list\n\n"
        "Conversation (Session {sid}):\n{conv}\n\n"
        'Respond with ONLY JSON: {{"facts": ["fact1", "fact2", ...]}}'
    )

    windows = []
    for s in samples:
        for sess in s["conversation"]:
            turns = [t for t in sess["dialogue"] if isinstance(t, dict)]
            for start in range(0, len(turns), stride):
                chunk = turns[start : start + window_size]
                lines = []
                for t in chunk:
                    sp = t.get("speaker", "?")
                    ut = t.get("utterance", "")
                    if ut.strip():
                        lines.append(f"{sp}: {ut}")
                if lines:
                    windows.append({
                        "sid": s["sample_id"],
                        "session_id": sess["session_id"],
                        "start": start,
                        "text": "\n".join(lines),
                    })

    print(f"  Windows to extract: {len(windows)}")
    all_facts = []
    for w in tqdm(windows, desc="  Extracting", unit="win"):
        try:
            p = prompt_tpl.format(sid=w["session_id"], conv=w["text"])
            resp = llm.invoke(p)
            m = re.search(r"\{.*\}", resp.content, re.DOTALL)
            parsed = json.loads(m.group()) if m else {}
            facts = [f.strip() for f in parsed.get("facts", []) if f.strip()]
        except Exception:
            facts = []
        for f in facts:
            all_facts.append({
                "content": f,
                "metadata": {
                    "sample_id": w["sid"],
                    "session_id": w["session_id"],
                    "source": "locomo_extracted",
                },
            })
    return all_facts


def ingest_facts(facts: list[dict], namespace: str = "locomo_extracted") -> int:
    """Embed and insert facts into archival memory."""
    conn = psycopg.connect(get_db_url())
    conn.execute("DELETE FROM archival_memory WHERE namespace = %s", (namespace,))
    conn.commit()
    register_vector(conn)

    embeddings = get_embeddings()
    total = 0
    for i in range(0, len(facts), BATCH_SIZE):
        batch = facts[i : i + BATCH_SIZE]
        vecs = embeddings.embed_documents([b["content"] for b in batch])
        for b, v in zip(batch, vecs):
            conn.execute(
                "INSERT INTO archival_memory (id, namespace, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (str(uuid.uuid4()), namespace, b["content"], json.dumps(b["metadata"]), Vector(v)),
            )
            total += 1
    conn.commit()
    conn.close()
    return total


def search(query: str, namespace: str, limit: int = 10, alpha: float = 0.7) -> list[dict]:
    emb = get_embeddings()
    qe = Vector(emb.embed_query(query))
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
        (alpha, qe, alpha, query, namespace, qe, limit),
    ).fetchall()
    conn.close()
    return [
        {"content": r[0], "metadata": r[1] if isinstance(r[1], dict) else json.loads(r[1]), "score": float(r[2])}
        for r in rows
    ]


def check(answer: str, results: list[dict], k: int) -> bool:
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


def evaluate(samples: list[dict], namespace: str, label: str) -> None:
    k_vals = [1, 3, 5, 10]
    results = {}
    t0 = time.time()
    for s in samples:
        for q in s["questions"]:
            qt = q["question_type"]
            if qt not in results:
                results[qt] = {k: {"hits": 0, "total": 0} for k in k_vals}
            for k in k_vals:
                results[qt][k]["total"] += 1
            sr = search(q["question"], namespace, limit=10)
            for k in k_vals:
                if check(q["answer"], sr, k):
                    results[qt][k]["hits"] += 1

    print(f"\n  {label}")
    print(f"  {'Type':<20}", end="")
    for k in k_vals:
        print(f"{'R@' + str(k):>8}", end="")
    print()
    print("  " + "-" * 56)
    th = {k: 0 for k in k_vals}
    tc = {k: 0 for k in k_vals}
    for qt in sorted(results):
        print(f"  {qt:<20}", end="")
        for k in k_vals:
            d = results[qt].get(k, {"hits": 0, "total": 1})
            r = d["hits"] / d["total"] if d["total"] > 0 else 0
            print(f"{r:>7.1%} ", end="")
            th[k] += d["hits"]
            tc[k] += d["total"]
        print()
    print("  " + "-" * 56)
    print(f"  {'Overall':<20}", end="")
    for k in k_vals:
        r = th[k] / tc[k] if tc[k] > 0 else 0
        print(f"{r:>7.1%} ", end="")
    print(f"\n  Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", default="data/locomo.json")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--stride", type=int, default=20, help="Window stride (larger = fewer LLM calls)")
    parser.add_argument("--window", type=int, default=10, help="Window size in turns")
    args = parser.parse_args()

    samples = load_samples(args.local, args.limit)
    total_q = sum(len(s["questions"]) for s in samples)
    print(f"Loaded {len(samples)} sample(s), {total_q} questions")

    # 1. Extract facts
    print("\n--- Extracting facts ---")
    facts = extract_facts(samples, window_size=args.window, stride=args.stride)
    print(f"  Total facts extracted: {len(facts)}")

    # 2. Ingest
    ns = "locomo_extracted"
    count = ingest_facts(facts, ns)
    print(f"  Ingested {count} facts into '{ns}'")

    # 3. Evaluate
    evaluate(samples, ns, f"Extracted Facts (stride={args.stride}, win={args.window})")
