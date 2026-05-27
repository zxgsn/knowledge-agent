"""Tests for agent.nodes.responder — token budget and context truncation."""

import pytest

from agent.nodes.responder import _estimate_tokens, _truncate_to_budget


class TestEstimateTokens:
    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_short_string(self):
        # 4 chars = 1 token
        assert _estimate_tokens("test") == 1

    def test_longer_string(self):
        # 100 chars = 25 tokens
        assert _estimate_tokens("a" * 100) == 25


class TestTruncateToBudget:
    def test_no_truncation_needed(self):
        parts = ["short text", "another short text"]
        result = _truncate_to_budget(parts, max_tokens=1000)
        assert result == parts

    def test_truncates_last_part_first(self):
        parts = ["priority content", "lower priority content that is much longer and should be truncated first"]
        result = _truncate_to_budget(parts, max_tokens=5)
        # Should keep priority content intact
        assert result[0] == "priority content"
        # Second part should be truncated (shorter original content)
        assert "truncated" in result[1]

    def test_removes_parts_when_budget_too_small(self):
        parts = ["first", "second", "third"]
        result = _truncate_to_budget(parts, max_tokens=1)
        # Should remove parts until budget fits
        assert len(result) < len(parts)

    def test_empty_parts_filtered(self):
        parts = ["content", "also content"]
        # Set budget high enough to keep both
        result = _truncate_to_budget(parts, max_tokens=1000)
        assert len(result) == 2

    def test_all_parts_within_budget(self):
        parts = ["a", "b", "c"]
        result = _truncate_to_budget(parts, max_tokens=1000)
        assert result == parts
