"""Tests for agent.utils — parse_json, get_env."""

import json

from agent.utils import get_env, parse_json


class TestParseJson:
    def test_valid_json(self):
        assert parse_json('{"mode": "chat", "need_recall": true}') == {
            "mode": "chat",
            "need_recall": True,
        }

    def test_json_in_markdown_fence(self):
        text = 'Some preamble\n```json\n{"mode": "research"}\n```\nTrailing'
        assert parse_json(text) == {"mode": "research"}

    def test_json_in_plain_fence(self):
        text = '```\n{"key": "value"}\n```'
        assert parse_json(text) == {"key": "value"}

    def test_json_with_extra_text_not_extracted(self):
        # parse_json only extracts from markdown fences, not from plain text
        text = 'Here is the result: {"is_sufficient": false, "reason": "gap"}'
        assert parse_json(text) == {}

    def test_invalid_json_returns_empty(self):
        assert parse_json("not json at all") == {}

    def test_empty_string_returns_empty(self):
        assert parse_json("") == {}

    def test_empty_json_object(self):
        assert parse_json("{}") == {}

    def test_json_array(self):
        assert parse_json("[1, 2, 3]") == [1, 2, 3]

    def test_nested_json(self):
        data = {"memory": [{"event": "ADD", "text": "hello"}]}
        assert parse_json(json.dumps(data)) == data

    def test_markdown_fence_with_lang_tag(self):
        text = "```json\n{\"a\": 1}\n```"
        assert parse_json(text) == {"a": 1}

    def test_malformed_json_in_fence(self):
        text = "```json\n{invalid}\n```"
        assert parse_json(text) == {}

    def test_whitespace_only(self):
        assert parse_json("   ") == {}


class TestGetEnv:
    def test_existing_var(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "hello")
        assert get_env("TEST_VAR") == "hello"

    def test_missing_var_returns_default(self):
        assert get_env("NONEXISTENT_VAR_XYZ_123", "fallback") == "fallback"

    def test_missing_var_no_default(self):
        assert get_env("NONEXISTENT_VAR_XYZ_123") == ""
