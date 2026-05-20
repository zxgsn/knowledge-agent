"""CLI interactive chat for the Knowledge Agent.

Usage:
    python -m examples.cli_chat
    python examples/cli_chat.py
    python examples/cli_chat.py --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import selectors
import sys

# Fix Windows: psycopg requires SelectorEventLoop, not the default ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Fix Windows terminal encoding for emoji/unicode output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import graph

load_dotenv()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Agent CLI")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--max-loops", type=int, default=2, help="Max research loops")
    args = parser.parse_args()

    print("=" * 60)
    print("  Knowledge Agent - Personal Research Assistant")
    print("=" * 60)
    print("Commands:")
    print("  /memory    - View core memory blocks")
    print("  /search    - Search archival knowledge base")
    print("  /ingest    - Ingest a document (URL or file path)")
    print("  /quit      - Exit")
    print("=" * 60)

    # Session state persists across turns
    session_state: dict = {
        "messages": [],
        "core_memory": {},
        "archival_results": [],
        "search_results": [],
        "sources": [],
        "research_loop_count": 0,
        "mode": "chat",
        "doc_source": "",
        "doc_source_type": "",
        "ingest_result": "",
        "search_query": [],
        "web_research_result": [],
        "sources_gathered": [],
        "is_sufficient": False,
        "knowledge_gap": "",
        "follow_up_queries": [],
        "max_research_loops": args.max_loops,
    }

    config = {
        "configurable": {
            "llm_model": args.model,
            "max_research_loops": args.max_loops,
        }
    }

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break

        if user_input.lower() == "/memory":
            _print_memory(session_state)
            continue

        if user_input.lower().startswith("/search "):
            query = user_input[8:].strip()
            session_state["messages"] = [HumanMessage(content=query)]
            session_state["mode"] = "research"
            session_state["doc_source"] = ""
            session_state["doc_source_type"] = ""
            session_state["ingest_result"] = ""
            session_state["search_query"] = []
            session_state["web_research_result"] = []
            session_state["sources"] = []
            session_state["archival_results"] = []
            session_state["is_sufficient"] = False
            session_state["knowledge_gap"] = ""
            session_state["follow_up_queries"] = []
            print(f"\n[Searching archival memory for: {query}]")
            print("\nAgent: ", end="", flush=True)
            try:
                result = await graph.ainvoke(session_state, config=config)
                for key in ("core_memory", "mode"):
                    if key in result:
                        session_state[key] = result[key]
                messages = result.get("messages", [])
                if messages:
                    content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
                    print(content)
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
            continue

        if user_input.lower().startswith("/ingest "):
            source = user_input[8:].strip()
            if source.startswith("http://") or source.startswith("https://"):
                session_state["doc_source"] = source
                session_state["doc_source_type"] = "url"
            else:
                if os.path.isfile(source):
                    with open(source, "r", encoding="utf-8") as f:
                        session_state["doc_source"] = f.read()
                    session_state["doc_source_type"] = "text"
                else:
                    session_state["doc_source"] = source
                    session_state["doc_source_type"] = "text"
            session_state["mode"] = "ingest"
            session_state["messages"] = [HumanMessage(content=f"Ingest: {source}")]
            print(f"\n[Ingesting: {source}]")
            print("\nAgent: ", end="", flush=True)
            try:
                result = await graph.ainvoke(session_state, config=config)
                ingest_result = result.get("ingest_result", "")
                print(ingest_result or "Ingestion complete.")
                if "core_memory" in result:
                    session_state["core_memory"] = result["core_memory"]
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
            continue

        # Normal chat/research mode
        session_state["messages"] = [HumanMessage(content=user_input)]
        session_state["mode"] = "chat"
        session_state["doc_source"] = ""
        session_state["doc_source_type"] = ""
        session_state["ingest_result"] = ""
        session_state["search_query"] = []
        session_state["web_research_result"] = []
        session_state["sources"] = []
        session_state["sources_gathered"] = []
        session_state["archival_results"] = []
        session_state["search_results"] = []
        session_state["is_sufficient"] = False
        session_state["knowledge_gap"] = ""
        session_state["follow_up_queries"] = []

        print("\nAgent: ", end="", flush=True)

        try:
            result = await graph.ainvoke(session_state, config=config)

            for key in ("core_memory", "mode"):
                if key in result:
                    session_state[key] = result[key]

            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                print(content)
            else:
                print("(no response)")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def _print_memory(state: dict) -> None:
    """Display current core memory blocks."""
    memory = state.get("core_memory", {})
    if not memory:
        print("\n[Core memory is empty]")
        return
    print("\n--- Core Memory ---")
    for label, value in memory.items():
        preview = value[:100] + "..." if len(value) > 100 else value
        print(f"\n[{label}] ({len(value)} chars)")
        print(f"  {preview}")
    print("---")


if __name__ == "__main__":
    asyncio.run(main())
