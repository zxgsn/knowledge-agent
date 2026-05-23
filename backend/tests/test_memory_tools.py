"""Tests for agent.memory.tools — core memory editing tools."""

from agent.memory.block import Block
from agent.memory.core_memory import CoreMemory
from agent.memory.tools import create_memory_tools


def _get_tool(tools, name):
    """Find a tool by name in the tools list."""
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool '{name}' not found")


class TestCoreMemoryReplace:
    def _setup(self):
        cm = CoreMemory()
        cm.update_block_value("human", "User is a developer named Alice.")
        tools = create_memory_tools(cm)
        return cm, _get_tool(tools, "core_memory_replace")

    def test_successful_replace(self):
        cm, tool = self._setup()
        result = tool.invoke({"label": "human", "old_string": "Alice", "new_string": "Bob"})
        assert "updated successfully" in result
        assert "Bob" in cm.get_block("human").value
        assert "Alice" not in cm.get_block("human").value

    def test_old_string_not_found(self):
        cm, tool = self._setup()
        result = tool.invoke({"label": "human", "old_string": "Charlie", "new_string": "Bob"})
        assert "Error" in result
        assert "not found" in result

    def test_old_string_multiple_matches(self):
        cm, tool = self._setup()
        cm.update_block_value("human", "a user and a user walk in.")
        result = tool.invoke({"label": "human", "old_string": "a user", "new_string": "the user"})
        assert "Error" in result
        assert "2 times" in result

    def test_nonexistent_block(self):
        cm, tool = self._setup()
        result = tool.invoke({"label": "nonexistent", "old_string": "x", "new_string": "y"})
        assert "Error" in result
        assert "No block" in result

    def test_read_only_block(self):
        cm = CoreMemory()
        cm.set_block(Block(label="ro", value="fixed", read_only=True))
        tools = create_memory_tools(cm)
        tool = _get_tool(tools, "core_memory_replace")
        result = tool.invoke({"label": "ro", "old_string": "fixed", "new_string": "new"})
        assert "Error" in result
        assert "read-only" in result

    def test_replace_exceeds_limit(self):
        cm = CoreMemory()
        cm.set_block(Block(label="tiny", value="hello", limit=10))
        tools = create_memory_tools(cm)
        tool = _get_tool(tools, "core_memory_replace")
        result = tool.invoke({"label": "tiny", "old_string": "hello", "new_string": "a" * 20})
        assert "Error" in result
        assert "exceeds limit" in result


class TestCoreMemoryInsert:
    def _setup(self):
        cm = CoreMemory()
        cm.update_block_value("human", "Line one\nLine two\nLine three")
        tools = create_memory_tools(cm)
        return cm, _get_tool(tools, "core_memory_insert")

    def test_append_to_end(self):
        cm, tool = self._setup()
        result = tool.invoke({"label": "human", "new_string": "Line four", "insert_line": -1})
        assert "updated" in result
        value = cm.get_block("human").value
        assert "Line four" in value
        assert value.endswith("Line four")

    def test_insert_at_beginning(self):
        cm, tool = self._setup()
        result = tool.invoke({"label": "human", "new_string": "Zero", "insert_line": 0})
        assert "updated" in result
        assert cm.get_block("human").value.startswith("Zero")

    def test_insert_at_middle(self):
        cm, tool = self._setup()
        result = tool.invoke({"label": "human", "new_string": "Inserted", "insert_line": 1})
        assert "updated" in result
        lines = cm.get_block("human").value.split("\n")
        assert lines[1] == "Inserted"

    def test_insert_out_of_range(self):
        cm, tool = self._setup()
        result = tool.invoke({"label": "human", "new_string": "bad", "insert_line": 10})
        assert "Error" in result
        assert "out of range" in result

    def test_insert_negative_out_of_range(self):
        cm, tool = self._setup()
        result = tool.invoke({"label": "human", "new_string": "bad", "insert_line": -5})
        assert "Error" in result

    def test_insert_nonexistent_block(self):
        cm, tool = self._setup()
        result = tool.invoke({"label": "missing", "new_string": "x", "insert_line": 0})
        assert "Error" in result

    def test_insert_read_only(self):
        cm = CoreMemory()
        cm.set_block(Block(label="ro", value="fixed", read_only=True))
        tools = create_memory_tools(cm)
        tool = _get_tool(tools, "core_memory_insert")
        result = tool.invoke({"label": "ro", "new_string": "new", "insert_line": -1})
        assert "Error" in result
        assert "read-only" in result

    def test_insert_multiline(self):
        cm, tool = self._setup()
        result = tool.invoke({"label": "human", "new_string": "A\nB\nC", "insert_line": 0})
        assert "updated" in result
        lines = cm.get_block("human").value.split("\n")
        assert lines[0] == "A"
        assert lines[1] == "B"
        assert lines[2] == "C"


class TestCoreMemoryRethink:
    def test_rewrite_existing_block(self):
        cm = CoreMemory()
        cm.update_block_value("human", "old content")
        tools = create_memory_tools(cm)
        tool = _get_tool(tools, "core_memory_rethink")
        result = tool.invoke({"label": "human", "new_memory": "completely new content"})
        assert "rewritten" in result
        assert cm.get_block("human").value == "completely new content"

    def test_rethink_creates_new_block(self):
        cm = CoreMemory()
        tools = create_memory_tools(cm)
        tool = _get_tool(tools, "core_memory_rethink")
        result = tool.invoke({"label": "new_block", "new_memory": "brand new"})
        assert "rewritten" in result
        assert cm.get_block("new_block").value == "brand new"
        assert "new_block" in cm.list_labels()

    def test_rethink_read_only(self):
        cm = CoreMemory()
        cm.set_block(Block(label="ro", value="fixed", read_only=True))
        tools = create_memory_tools(cm)
        tool = _get_tool(tools, "core_memory_rethink")
        result = tool.invoke({"label": "ro", "new_memory": "new"})
        assert "Error" in result
        assert "read-only" in result

    def test_rethink_exceeds_limit(self):
        cm = CoreMemory()
        cm.set_block(Block(label="tiny", value="x", limit=10))
        tools = create_memory_tools(cm)
        tool = _get_tool(tools, "core_memory_rethink")
        result = tool.invoke({"label": "tiny", "new_memory": "a" * 20})
        assert "Error" in result


class TestCoreMemoryView:
    def test_view_specific_block(self):
        cm = CoreMemory()
        cm.update_block_value("human", "Test user info")
        tools = create_memory_tools(cm)
        tool = _get_tool(tools, "core_memory_view")
        result = tool.invoke({"label": "human"})
        assert "<human>" in result
        assert "Test user info" in result
        assert "description:" in result
        assert "chars:" in result

    def test_view_all_blocks(self):
        cm = CoreMemory()
        tools = create_memory_tools(cm)
        tool = _get_tool(tools, "core_memory_view")
        result = tool.invoke({"label": ""})
        assert "<memory_blocks>" in result
        assert "<persona>" in result
        assert "<human>" in result

    def test_view_nonexistent_block(self):
        cm = CoreMemory()
        tools = create_memory_tools(cm)
        tool = _get_tool(tools, "core_memory_view")
        result = tool.invoke({"label": "nonexistent"})
        assert "Error" in result


class TestCreateMemoryTools:
    def test_returns_core_tools_only(self):
        cm = CoreMemory()
        tools = create_memory_tools(cm, enable_archival=False)
        names = [t.name for t in tools]
        assert "core_memory_replace" in names
        assert "core_memory_insert" in names
        assert "core_memory_rethink" in names
        assert "core_memory_view" in names
        assert "archival_memory_search" not in names

    def test_returns_archival_tools_when_enabled(self):
        cm = CoreMemory()
        tools = create_memory_tools(cm, enable_archival=True)
        names = [t.name for t in tools]
        assert "archival_memory_search" in names
        assert "archival_memory_save" in names
        assert "ingest_document" in names

    def test_tools_count(self):
        cm = CoreMemory()
        assert len(create_memory_tools(cm, enable_archival=False)) == 4
        assert len(create_memory_tools(cm, enable_archival=True)) == 7
