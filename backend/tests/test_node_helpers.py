"""Tests for responder formatting helpers and memory_manager topic extraction."""

from unittest.mock import patch

from agent.nodes.responder import _format_archival_results, _format_recall_results


class TestFormatArchivalResults:
    def test_empty(self):
        assert _format_archival_results([]) == ""

    def test_high_score_included(self):
        results = [{"content": "fact A", "score": 0.8}]
        out = _format_archival_results(results)
        assert "fact A" in out
        assert "0.80" in out

    def test_low_score_included(self):
        results = [{"content": "noise", "score": 0.1}]
        out = _format_archival_results(results)
        assert "noise" in out
        assert "0.10" in out

    def test_mixed_scores(self):
        results = [
            {"content": "good", "score": 0.5},
            {"content": "low", "score": 0.2},
        ]
        out = _format_archival_results(results)
        assert "good" in out
        assert "low" in out

    def test_multiple_results(self):
        results = [
            {"content": "A", "score": 0.9},
            {"content": "B", "score": 0.7},
            {"content": "C", "score": 0.4},
        ]
        out = _format_archival_results(results)
        assert "A" in out
        assert "B" in out
        assert "C" in out


class TestFormatRecallResults:
    def test_empty(self):
        assert _format_recall_results([]) == ""

    def test_high_score_included(self):
        results = [{"content": "past msg", "role": "user", "score": 0.8}]
        out = _format_recall_results(results)
        assert "past msg" in out
        assert "user" in out

    def test_low_score_included(self):
        results = [{"content": "noise", "role": "user", "score": 0.1}]
        out = _format_recall_results(results)
        assert "noise" in out
        assert "0.10" in out

    def test_long_content_truncated(self):
        results = [{"content": "x" * 500, "role": "assistant", "score": 0.9}]
        out = _format_recall_results(results)
        assert "..." in out
        assert len(out) < 500

    def test_role_displayed(self):
        results = [
            {"content": "msg1", "role": "user", "score": 0.9},
            {"content": "msg2", "role": "assistant", "score": 0.8},
        ]
        out = _format_recall_results(results)
        assert "user" in out
        assert "assistant" in out


class TestGetResearchTopic:
    def _get_fn(self):
        from agent.nodes.memory_manager import _get_research_topic
        return _get_research_topic

    def test_single_human_message(self):
        from langchain_core.messages import HumanMessage
        fn = self._get_fn()
        msgs = [HumanMessage(content="What is Python?")]
        assert fn(msgs) == "What is Python?"

    def test_multiple_human_messages(self):
        from langchain_core.messages import HumanMessage
        fn = self._get_fn()
        msgs = [
            HumanMessage(content="Topic A"),
            HumanMessage(content="Topic B"),
        ]
        result = fn(msgs)
        assert "Topic A" in result
        assert "Topic B" in result

    def test_no_human_messages(self):
        from langchain_core.messages import AIMessage
        fn = self._get_fn()
        msgs = [AIMessage(content="I am AI")]
        assert fn(msgs) == ""

    def test_empty_list(self):
        fn = self._get_fn()
        assert fn([]) == ""
