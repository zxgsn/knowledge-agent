from pydantic import BaseModel, Field


class Block(BaseModel):
    """A named, editable memory block that lives inside the system prompt.

    Inspired by Letta's Block design. Blocks are addressed by `label`
    and rendered into the system prompt via Memory.compile().
    """

    label: str = Field(description="Unique identifier, e.g. 'persona', 'human'")
    value: str = Field(default="", description="The text content of this block")
    limit: int = Field(default=5000, description="Character limit for value")
    description: str = Field(
        default="", description="Explains how this block should influence agent behavior"
    )
    read_only: bool = Field(
        default=False, description="If True, the agent cannot edit this block"
    )

    @property
    def chars_current(self) -> int:
        return len(self.value)

    def is_within_limit(self, text: str | None = None) -> bool:
        check = text if text is not None else self.value
        return len(check) <= self.limit


# Default block templates
DEFAULT_BLOCKS: list[Block] = [
    Block(
        label="persona",
        value="I am a knowledge agent with persistent memory. I can research topics via web search, ingest documents into my knowledge base, and recall past research. My archival memory stores knowledge as vector embeddings in PostgreSQL + pgvector, retrievable via semantic search.",
        description="The agent's personality and role definition.",
        read_only=False,
    ),
    Block(
        label="human",
        value="",
        description="Information about the user you are talking to.",
        read_only=False,
    ),
    Block(
        label="knowledge_focus",
        value="",
        description="Current research topics and areas of interest.",
        read_only=False,
    ),
]
