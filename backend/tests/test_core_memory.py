"""Tests for agent.memory.core_memory — CoreMemory and Block."""

import pytest

from agent.memory.block import Block
from agent.memory.core_memory import CoreMemory


class TestBlock:
    def test_default_values(self):
        b = Block(label="test")
        assert b.value == ""
        assert b.limit == 5000
        assert b.read_only is False
        assert b.description == ""

    def test_chars_current(self):
        b = Block(label="test", value="hello world")
        assert b.chars_current == 11

    def test_chars_current_empty(self):
        b = Block(label="test")
        assert b.chars_current == 0

    def test_is_within_limit(self):
        b = Block(label="test", limit=10)
        assert b.is_within_limit("short") is True
        assert b.is_within_limit("a" * 11) is False
        assert b.is_within_limit("a" * 10) is True

    def test_is_within_limit_uses_value_by_default(self):
        b = Block(label="test", value="hello", limit=10)
        assert b.is_within_limit() is True

    def test_read_only(self):
        b = Block(label="test", read_only=True)
        assert b.read_only is True


class TestCoreMemory:
    def test_default_blocks(self):
        cm = CoreMemory()
        labels = cm.list_labels()
        assert "persona" in labels
        assert "human" in labels
        assert "knowledge_focus" in labels

    def test_get_block(self):
        cm = CoreMemory()
        block = cm.get_block("persona")
        assert block.label == "persona"
        assert len(block.value) > 0

    def test_get_block_missing_raises(self):
        cm = CoreMemory()
        with pytest.raises(KeyError):
            cm.get_block("nonexistent")

    def test_set_block_new(self):
        cm = CoreMemory()
        new_block = Block(label="custom", value="custom value")
        cm.set_block(new_block)
        assert "custom" in cm.list_labels()
        assert cm.get_block("custom").value == "custom value"

    def test_set_block_replaces_existing(self):
        cm = CoreMemory()
        cm.set_block(Block(label="persona", value="new persona"))
        assert cm.get_block("persona").value == "new persona"

    def test_update_block_value(self):
        cm = CoreMemory()
        cm.update_block_value("human", "User is a developer")
        assert cm.get_block("human").value == "User is a developer"

    def test_update_block_value_exceeds_limit(self):
        cm = CoreMemory()
        cm.get_block("human").limit = 10
        with pytest.raises(ValueError, match="exceeds limit"):
            cm.update_block_value("human", "a" * 11)

    def test_update_block_value_missing_raises(self):
        cm = CoreMemory()
        with pytest.raises(KeyError):
            cm.update_block_value("nonexistent", "value")

    def test_compile(self):
        cm = CoreMemory()
        xml = cm.compile()
        assert "<memory_blocks>" in xml
        assert "</memory_blocks>" in xml
        assert "<persona>" in xml
        assert "<human>" in xml

    def test_compile_empty(self):
        cm = CoreMemory(blocks=[])
        assert cm.compile() == ""

    def test_to_dict(self):
        cm = CoreMemory()
        d = cm.to_dict()
        assert isinstance(d, dict)
        assert "persona" in d
        assert isinstance(d["persona"], str)

    def test_from_dict(self):
        data = {"persona": "I am a bot", "human": "User is Alice"}
        cm = CoreMemory.from_dict(data)
        assert cm.get_block("persona").value == "I am a bot"
        assert cm.get_block("human").value == "User is Alice"

    def test_from_dict_creates_unknown_blocks(self):
        data = {"custom_key": "custom value"}
        cm = CoreMemory.from_dict(data)
        assert cm.get_block("custom_key").value == "custom value"

    def test_from_dict_empty(self):
        cm = CoreMemory.from_dict({})
        # Should fall back to defaults
        assert "persona" in cm.list_labels()

    def test_to_dict_from_dict_roundtrip(self):
        cm = CoreMemory()
        cm.update_block_value("human", "Test user")
        d = cm.to_dict()
        cm2 = CoreMemory.from_dict(d)
        assert cm2.get_block("human").value == "Test user"

    def test_list_labels(self):
        cm = CoreMemory()
        labels = cm.list_labels()
        assert isinstance(labels, list)
        assert len(labels) >= 3

    def test_blocks_property(self):
        cm = CoreMemory()
        blocks = cm.blocks
        assert isinstance(blocks, list)
        assert all(isinstance(b, Block) for b in blocks)

    def test_get_blocks_needing_compression(self):
        cm = CoreMemory()
        # Default blocks are mostly empty, should not need compression
        assert cm.get_blocks_needing_compression() == []

    def test_get_blocks_needing_compression_when_full(self):
        cm = CoreMemory()
        cm.get_block("human").limit = 100
        cm.update_block_value("human", "a" * 85)  # 85% full
        needing = cm.get_blocks_needing_compression()
        assert "human" in needing

    def test_read_only_block_not_compressed(self):
        cm = CoreMemory()
        block = Block(label="readonly", value="a" * 100, limit=100, read_only=True)
        cm.set_block(block)
        needing = cm.get_blocks_needing_compression()
        assert "readonly" not in needing


class TestEditHistory:
    def test_update_records_history(self):
        cm = CoreMemory()
        cm.update_block_value("human", "Alice")
        history = cm.get_edit_history()
        assert len(history) == 1
        assert history[0]["label"] == "human"
        assert history[0]["old_value"] == ""
        assert history[0]["new_value"] == "Alice"
        assert history[0]["operation"] == "update"

    def test_multiple_edits_recorded(self):
        cm = CoreMemory()
        cm.update_block_value("human", "Alice")
        cm.update_block_value("human", "Bob")
        assert len(cm.get_edit_history()) == 2

    def test_update_without_record(self):
        cm = CoreMemory()
        cm.update_block_value("human", "Alice", record=False)
        assert len(cm.get_edit_history()) == 0

    def test_history_bounded_at_50(self):
        cm = CoreMemory()
        for i in range(60):
            cm.update_block_value("human", f"val_{i}")
        assert len(cm.get_edit_history()) == 50

    def test_record_edit_method(self):
        cm = CoreMemory()
        cm._record_edit("human", "old", "new", "create")
        history = cm.get_edit_history()
        assert history[0]["operation"] == "create"


class TestUndo:
    def test_undo_update(self):
        cm = CoreMemory()
        cm.update_block_value("human", "Alice")
        cm.update_block_value("human", "Bob")
        assert cm.get_block("human").value == "Bob"

        entry = cm.undo_last_edit()
        assert entry["label"] == "human"
        assert entry["operation"] == "update"
        assert cm.get_block("human").value == "Alice"

    def test_undo_multiple(self):
        cm = CoreMemory()
        cm.update_block_value("human", "Alice")
        cm.update_block_value("human", "Bob")
        cm.update_block_value("human", "Charlie")

        cm.undo_last_edit()
        assert cm.get_block("human").value == "Bob"
        cm.undo_last_edit()
        assert cm.get_block("human").value == "Alice"

    def test_undo_empty_history(self):
        cm = CoreMemory()
        assert cm.undo_last_edit() is None

    def test_undo_create_removes_block(self):
        cm = CoreMemory()
        cm._record_edit("new_block", "", "some value", "create")
        cm.set_block(Block(label="new_block", value="some value"))
        assert "new_block" in cm.list_labels()

        entry = cm.undo_last_edit()
        assert entry["operation"] == "create"
        assert "new_block" not in cm.list_labels()

    def test_get_edit_history_returns_copy(self):
        cm = CoreMemory()
        cm.update_block_value("human", "Alice")
        h = cm.get_edit_history()
        h.clear()
        assert len(cm.get_edit_history()) == 1


class TestToDictWithHistory:
    def test_includes_blocks_and_history(self):
        cm = CoreMemory()
        cm.update_block_value("human", "Alice")
        d = cm.to_dict_with_history()
        assert "blocks" in d
        assert "edit_history" in d
        assert d["blocks"]["human"] == "Alice"
        assert len(d["edit_history"]) == 1

    def test_roundtrip_with_history(self):
        cm = CoreMemory()
        cm.update_block_value("human", "Alice")
        cm.update_block_value("human", "Bob")
        d = cm.to_dict_with_history()

        cm2 = CoreMemory.from_dict(d)
        assert cm2.get_block("human").value == "Bob"
        assert len(cm2.get_edit_history()) == 2

    def test_from_dict_backward_compatible(self):
        """Old format (plain dict) should still work."""
        data = {"persona": "I am a bot", "human": "Alice"}
        cm = CoreMemory.from_dict(data)
        assert cm.get_block("human").value == "Alice"
        assert len(cm.get_edit_history()) == 0
