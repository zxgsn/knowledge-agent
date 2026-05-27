"""Tests for agent.nodes.memory_pipeline — summarization and pipeline logic."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage


def _run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSummarizeOldMessages:
    def test_short_conversation_unchanged(self):
        from agent.nodes.memory_pipeline import summarize_old_messages

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]
        llm = AsyncMock()

        result = _run_async(summarize_old_messages(messages, llm))
        assert result == messages
        llm.ainvoke.assert_not_called()

    def test_long_conversation_summarized(self):
        from agent.nodes.memory_pipeline import summarize_old_messages, SUMMARIZE_THRESHOLD

        # Create messages exceeding threshold
        messages = []
        for i in range(SUMMARIZE_THRESHOLD + 5):
            if i % 2 == 0:
                messages.append(HumanMessage(content=f"User message {i}"))
            else:
                messages.append(AIMessage(content=f"Assistant message {i}"))

        llm = AsyncMock()
        llm.ainvoke.return_value = MagicMock(content="Summary of conversation")

        result = _run_async(summarize_old_messages(messages, llm, keep_recent=10))
        assert len(result) < len(messages)
        assert "[Conversation Summary]" in result[0].content
        llm.ainvoke.assert_called_once()

    def test_llm_failure_returns_original(self):
        from agent.nodes.memory_pipeline import summarize_old_messages, SUMMARIZE_THRESHOLD

        messages = [HumanMessage(content=f"msg {i}") for i in range(SUMMARIZE_THRESHOLD + 1)]

        llm = AsyncMock()
        llm.ainvoke.side_effect = RuntimeError("LLM failed")

        result = _run_async(summarize_old_messages(messages, llm))
        assert result == messages

    def test_preserves_recent_messages(self):
        from agent.nodes.memory_pipeline import summarize_old_messages, SUMMARIZE_THRESHOLD

        messages = []
        for i in range(SUMMARIZE_THRESHOLD + 10):
            messages.append(HumanMessage(content=f"msg {i}"))

        llm = AsyncMock()
        llm.ainvoke.return_value = MagicMock(content="Summary")

        keep_recent = 5
        result = _run_async(summarize_old_messages(messages, llm, keep_recent=keep_recent))

        # Last keep_recent messages should be preserved
        for i in range(keep_recent):
            assert result[-(i + 1)].content == messages[-(i + 1)].content
