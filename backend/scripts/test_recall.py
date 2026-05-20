"""Recall rate comparison: pure vector vs hybrid search.

Usage: python scripts/test_recall.py
"""
from __future__ import annotations

import asyncio
import io
import os
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


# --- Test documents to ingest ---
TEST_DOCS = [
    {
        "content": (
            "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models "
            "by retrieving relevant documents from an external knowledge base before generating a response. "
            "RAG systems typically use vector embeddings to find semantically similar chunks, then pass "
            "those chunks as context to the LLM. This approach reduces hallucination and keeps responses "
            "grounded in factual data."
        ),
        "metadata": {"source": "rag_overview", "topic": "RAG"},
    },
    {
        "content": (
            "PostgreSQL pgvector is an extension that adds vector similarity search capabilities to PostgreSQL. "
            "It supports cosine distance (<=>), L2 distance (<->), and inner product (<#>) operators. "
            "Index types include IVFFlat for approximate nearest neighbor search and HNSW for faster queries. "
            "The typical embedding dimension for modern models is 768 (BGE) or 1024 (DashScope text-embedding-v3)."
        ),
        "metadata": {"source": "pgvector_docs", "topic": "pgvector"},
    },
    {
        "content": (
            "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
            "It extends LangChain with cyclic graph capabilities, allowing complex agent workflows "
            "with conditional routing, parallel execution, and persistent state. Key concepts include "
            "StateGraph, nodes, edges, conditional edges, and the Send API for fan-out patterns."
        ),
        "metadata": {"source": "langgraph_intro", "topic": "LangGraph"},
    },
    {
        "content": (
            "量子计算利用量子力学原理（叠加态和纠缠态）来处理信息。与经典计算机使用比特（0或1）不同，"
            "量子计算机使用量子比特（qubit），可以同时处于0和1的叠加态。主要的量子计算技术路线包括"
            "超导量子比特、离子阱、光量子和拓扑量子计算。Google的Sycamore处理器在2019年实现了"
            "量子优越性，用200秒完成了经典超级计算机需要1万年才能完成的计算任务。"
        ),
        "metadata": {"source": "quantum_computing_zh", "topic": "quantum computing"},
    },
    {
        "content": (
            "Transformer architecture, introduced in the paper 'Attention Is All You Need' (Vaswani et al., 2017), "
            "revolutionized natural language processing. The key innovation is the self-attention mechanism, "
            "which allows the model to weigh the importance of different parts of the input sequence when "
            "processing each element. Transformers use multi-head attention, positional encoding, feed-forward "
            "networks, and layer normalization. GPT, BERT, and T5 are all based on the Transformer architecture."
        ),
        "metadata": {"source": "transformer_architecture", "topic": "Transformer"},
    },
    {
        "content": (
            "Letta (formerly MemGPT) is a framework for building LLM agents with long-term memory. "
            "It implements a tiered memory system inspired by operating system memory hierarchies: "
            "Core Memory (in-context, fast access), Recall Memory (conversation history, searchable), "
            "and Archival Memory (long-term storage, semantic search). Agents can autonomously manage "
            "their memory by reading, writing, and compressing memory blocks."
        ),
        "metadata": {"source": "letta_memory", "topic": "Letta"},
    },
    {
        "content": (
            "BM25 (Best Matching 25) is a ranking function used by search engines to estimate the relevance "
            "of documents to a given search query. It is based on the probabilistic retrieval model and "
            "extends TF-IDF by considering term frequency saturation and document length normalization. "
            "BM25 is widely used in information retrieval systems, including Elasticsearch and Lucene."
        ),
        "metadata": {"source": "bm25_explained", "topic": "BM25"},
    },
    {
        "content": (
            "混合检索（Hybrid Search）结合了向量语义检索和关键词检索（如BM25）的优势。"
            "向量检索擅长理解语义相似性，但对精确关键词匹配不敏感；BM25擅长精确匹配，"
            "但无法理解同义词和语义关系。混合检索通过加权组合两种分数来获得更好的召回率。"
            "典型的权重设置是向量0.7 + BM25 0.3，可根据具体场景调整。"
        ),
        "metadata": {"source": "hybrid_search_zh", "topic": "hybrid search"},
    },
    {
        "content": (
            "Core Memory auto-compression is a technique where an LLM agent automatically summarizes "
            "its core memory blocks when they approach their character limit. Instead of discarding "
            "information or throwing errors, the agent uses a focused LLM call to distill the content "
            "to a target size (e.g., 60% of the limit) while preserving all key facts. This is inspired "
            "by Letta's sleeptime memory management pattern."
        ),
        "metadata": {"source": "memory_compression", "topic": "memory compression"},
    },
    {
        "content": (
            "DashScope text-embedding-v3 is Alibaba Cloud's embedding model that produces 1024-dimensional "
            "vectors. It supports both Chinese and English text, with a maximum input of 8192 tokens. "
            "The model achieves strong performance on the MTEB and C-MTEB benchmarks. It can be accessed "
            "via the DashScope API or through the langchain-community DashScopeEmbeddings integration."
        ),
        "metadata": {"source": "dashscope_embedding", "topic": "DashScope"},
    },
]

# --- Test queries ---
# Easy queries (keyword-rich, close to original text)
EASY_QUERIES = [
    {"query": "RAG retrieval augmented generation hallucination", "expected_source": "rag_overview"},
    {"query": "pgvector PostgreSQL cosine distance IVFFlat HNSW", "expected_source": "pgvector_docs"},
    {"query": "LangGraph stateful multi-actor cyclic graph", "expected_source": "langgraph_intro"},
    {"query": "量子比特 超导 量子优越性 Google Sycamore", "expected_source": "quantum_computing_zh"},
    {"query": "self-attention mechanism Transformer Vaswani 2017", "expected_source": "transformer_architecture"},
    {"query": "Letta MemGPT tiered memory core recall archival", "expected_source": "letta_memory"},
    {"query": "BM25 term frequency saturation document length", "expected_source": "bm25_explained"},
    {"query": "混合检索 向量 BM25 加权 召回率", "expected_source": "hybrid_search_zh"},
    {"query": "auto-compression core memory character limit sleeptime", "expected_source": "memory_compression"},
    {"query": "DashScope text-embedding-v3 1024 dimensions MTEB", "expected_source": "dashscope_embedding"},
]

# Hard queries: paraphrased, short, or using different vocabulary
HARD_QUERIES = [
    {"query": "how to reduce LLM making things up", "expected_source": "rag_overview"},
    {"query": "vector search in Postgres database", "expected_source": "pgvector_docs"},
    {"query": "building agents with loops and branching", "expected_source": "langgraph_intro"},
    {"query": "qubit superconducting quantum supremacy processor", "expected_source": "quantum_computing_zh"},
    {"query": "attention mechanism NLP paper 2017", "expected_source": "transformer_architecture"},
    {"query": "operating system inspired AI memory hierarchy", "expected_source": "letta_memory"},
    {"query": "search ranking algorithm Elasticsearch Lucene", "expected_source": "bm25_explained"},
    {"query": "语义检索和关键词检索结合", "expected_source": "hybrid_search_zh"},
    {"query": "summarize memory when approaching limit", "expected_source": "memory_compression"},
    {"query": "Alibaba embedding model Chinese English bilingual", "expected_source": "dashscope_embedding"},
]


def ingest_test_docs() -> None:
    """Ingest test documents into the database."""
    embeddings = get_embeddings()
    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    # Clear old test data
    sources = [d["metadata"]["source"] for d in TEST_DOCS]
    conn.execute("DELETE FROM archival_memory WHERE metadata->>'source' = ANY(%s)", (sources,))

    contents = [d["content"] for d in TEST_DOCS]
    vectors = embeddings.embed_documents(contents)
    now = datetime.now(timezone.utc).isoformat()

    for doc, vec in zip(TEST_DOCS, vectors):
        entry_id = str(uuid.uuid4())
        meta = doc["metadata"]
        meta["timestamp"] = now
        conn.execute(
            "INSERT INTO archival_memory (id, namespace, content, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s)",
            (entry_id, "test", doc["content"], __import__("json").dumps(meta), Vector(vec)),
        )

    conn.commit()
    conn.close()
    print(f"Ingested {len(TEST_DOCS)} test documents.")


def search_vector_only(query: str, limit: int = 5) -> list[dict]:
    """Pure vector search (no BM25)."""
    embeddings = get_embeddings()
    query_embedding = Vector(embeddings.embed_query(query))
    conn = psycopg.connect(get_db_url())
    register_vector(conn)
    rows = conn.execute(
        """
        SELECT content, metadata, 1 - (embedding <=> %s::vector) AS score
        FROM archival_memory
        WHERE namespace = 'test'
        ORDER BY score DESC LIMIT %s
        """,
        (query_embedding, limit),
    ).fetchall()
    conn.close()
    return [
        {
            "content": r[0],
            "source": (r[1] if isinstance(r[1], dict) else __import__("json").loads(r[1])).get("source", ""),
            "score": float(r[2]),
        }
        for r in rows
    ]


def search_hybrid(query: str, limit: int = 5, alpha: float = 0.7) -> list[dict]:
    """Hybrid search: vector + BM25."""
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
        WHERE namespace = 'test'
          AND (content_tsv @@ plainto_tsquery('simple', %s)
               OR 1 - (embedding <=> %s::vector) > 0.2)
        ORDER BY score DESC
        LIMIT %s
        """,
        (alpha, query_embedding, alpha, query, query, query_embedding, limit),
    ).fetchall()
    conn.close()
    return [
        {
            "content": r[0],
            "source": (r[1] if isinstance(r[1], dict) else __import__("json").loads(r[1])).get("source", ""),
            "score": float(r[2]),
        }
        for r in rows
    ]


def run_test_for_queries(queries: list[dict], label: str) -> tuple[int, int]:
    """Run recall test for a set of queries. Returns (vector_hits, hybrid_hits)."""
    recall_vector = 0
    recall_hybrid = 0
    total = len(queries)

    print(f"\n  {'Query':<42} {'Vec':>4} {'Hyb':>4}")
    print("  " + "-" * 55)

    for tq in queries:
        query = tq["query"]
        expected = tq["expected_source"]

        vec_results = search_vector_only(query, limit=5)
        hyb_results = search_hybrid(query, limit=5)

        vec_hit = any(r["source"] == expected for r in vec_results)
        hyb_hit = any(r["source"] == expected for r in hyb_results)

        recall_vector += int(vec_hit)
        recall_hybrid += int(hyb_hit)

        q_display = query[:39] + "..." if len(query) > 39 else query
        print(f"  {q_display:<42} {'Y' if vec_hit else '-':>4} {'Y' if hyb_hit else '-':>4}")

    print("  " + "-" * 55)
    print(f"  Recall@5:  {recall_vector}/{total} = {recall_vector/total:.1%}   "
          f"{recall_hybrid}/{total} = {recall_hybrid/total:.1%}")

    return recall_vector, recall_hybrid


def run_test() -> None:
    """Run recall comparison test."""
    print("=" * 60)
    print("  Recall Rate Comparison: Vector vs Hybrid Search")
    print("=" * 60)

    print("\n--- Easy queries (close to original text) ---")
    easy_v, easy_h = run_test_for_queries(EASY_QUERIES, "Easy")

    print("\n--- Hard queries (paraphrased / different vocabulary) ---")
    hard_v, hard_h = run_test_for_queries(HARD_QUERIES, "Hard")

    total = len(EASY_QUERIES) + len(HARD_QUERIES)
    all_v = easy_v + hard_v
    all_h = easy_h + hard_h

    print("\n" + "=" * 60)
    print(f"  Overall Recall@5:  Vector {all_v}/{total} = {all_v/total:.1%}   "
          f"Hybrid {all_h}/{total} = {all_h/total:.1%}")

    if all_h > all_v:
        print(f"  Hybrid search improved overall recall by +{(all_h - all_v)/total:.1%}")
    elif all_h == all_v:
        print(f"  Both methods achieved the same overall recall rate.")
    else:
        print(f"  Vector search performed better overall by +{(all_v - all_h)/total:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    ingest_test_docs()
    run_test()
