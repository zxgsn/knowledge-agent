from __future__ import annotations

from agent.memory.block import Block, DEFAULT_BLOCKS


class CoreMemory:
    """In-context memory container holding a list of Blocks.

    The compile() method renders all blocks into an XML string that gets
    injected into the system prompt before each LLM call.

    Inspired by Letta's Memory class but simplified for LangGraph usage.
    """

    def __init__(self, blocks: list[Block] | None = None):
        self._blocks: list[Block] = blocks if blocks is not None else [
            b.model_copy(deep=True) for b in DEFAULT_BLOCKS
        ]
        self._edit_history: list[dict] = []

    @property
    def blocks(self) -> list[Block]:
        return self._blocks

    def get_block(self, label: str) -> Block:
        for b in self._blocks:
            if b.label == label:
                return b
        raise KeyError(f"No block with label '{label}'")

    def set_block(self, block: Block) -> None:
        for i, b in enumerate(self._blocks):
            if b.label == block.label:
                self._blocks[i] = block
                return
        self._blocks.append(block)

    def _record_edit(self, label: str, old_value: str, new_value: str, operation: str) -> None:
        """Record an edit in the history stack."""
        self._edit_history.append({
            "label": label,
            "old_value": old_value,
            "new_value": new_value,
            "operation": operation,
        })
        # Keep history bounded
        if len(self._edit_history) > 50:
            self._edit_history = self._edit_history[-50:]

    def update_block_value(self, label: str, value: str, record: bool = True) -> None:
        block = self.get_block(label)
        if len(value) > block.limit:
            raise ValueError(
                f"Value for block '{label}' exceeds limit: {len(value)} > {block.limit}"
            )
        if record:
            self._record_edit(label, block.value, value, "update")
        block.value = value

    def undo_last_edit(self) -> dict | None:
        """Undo the last edit. Returns the undone edit info, or None if no history."""
        if not self._edit_history:
            return None
        entry = self._edit_history.pop()
        label = entry["label"]
        old_value = entry["old_value"]
        if entry["operation"] == "create":
            # Block was created — remove it entirely
            self._blocks = [b for b in self._blocks if b.label != label]
        else:
            try:
                block = self.get_block(label)
                block.value = old_value
            except KeyError:
                pass
        return entry

    def get_edit_history(self) -> list[dict]:
        """Return the edit history (read-only copy)."""
        return list(self._edit_history)

    async def compress_block(self, label: str, llm) -> bool:
        """Compress a block that's approaching its character limit.

        Uses LLM to distill the content to ~60% of the limit.
        Returns True if compression was performed.
        """
        block = self.get_block(label)
        if block.read_only:
            return False
        usage_ratio = block.chars_current / block.limit
        if usage_ratio < 0.8:
            return False

        target_chars = int(block.limit * 0.6)
        prompt = (
            f"Compress the following memory block content to under {target_chars} characters. "
            f"Preserve ALL key facts, names, dates, and relationships. "
            f"Remove redundancy and verbose phrasing. Output ONLY the compressed text.\n\n"
            f"{block.value}"
        )
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        self.update_block_value(label, response.content, record=True)
        return True

    def get_blocks_needing_compression(self) -> list[str]:
        """Return labels of non-read-only blocks that are >= 80% full."""
        return [
            b.label for b in self._blocks
            if not b.read_only and b.chars_current / b.limit >= 0.8
        ]

    def list_labels(self) -> list[str]:
        return [b.label for b in self._blocks]

    def compile(self) -> str:
        """Render all blocks into XML for system prompt injection."""
        if not self._blocks:
            return ""

        parts = ["<memory_blocks>"]
        parts.append(
            "The following memory blocks are currently in your core memory:\n"
        )
        for block in self._blocks:
            parts.append(f"<{block.label}>")
            if block.description:
                parts.append(f"<description>{block.description}</description>")
            parts.append(
                f"<metadata>chars_current={block.chars_current}, "
                f"chars_limit={block.limit}, "
                f"read_only={block.read_only}</metadata>"
            )
            parts.append(f"<value>\n{block.value}\n</value>")
            parts.append(f"</{block.label}>\n")
        parts.append("</memory_blocks>")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, str]:
        """Serialize blocks to a simple dict for state storage."""
        return {b.label: b.value for b in self._blocks}

    def to_dict_with_history(self) -> dict:
        """Serialize blocks and edit history for state storage."""
        return {
            "blocks": {b.label: b.value for b in self._blocks},
            "edit_history": self._edit_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str], templates: list[Block] | None = None) -> CoreMemory:
        """Reconstruct CoreMemory from a dict of {label: value}.

        Templates provide the metadata (description, limit, read_only).
        If a label exists in data but not in templates, a default Block is created.
        """
        # Handle new format with history
        if "blocks" in data and isinstance(data["blocks"], dict):
            block_data = data["blocks"]
            history = data.get("edit_history", [])
        else:
            block_data = data
            history = []

        template_map = {}
        source = templates if templates is not None else DEFAULT_BLOCKS
        for t in source:
            template_map[t.label] = t

        blocks: list[Block] = []
        for label, value in block_data.items():
            if label in template_map:
                block = template_map[label].model_copy(update={"value": value})
            else:
                block = Block(label=label, value=value)
            blocks.append(block)
        cm = cls(blocks=blocks or None)
        cm._edit_history = history
        return cm
