"""Document ingestion pipeline: extract text → chunk → embed → store."""

from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup


@dataclass
class Chunk:
    """A text chunk with metadata."""

    content: str
    index: int
    source: str = ""
    source_type: str = ""  # "pdf", "url", "text"
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.source}_{self.index}"


def extract_text_from_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(pages)


def extract_text_from_url(url: str) -> tuple[str, str]:
    """Fetch a URL and extract readable text content.

    Returns (title, text).
    """
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    # Try to find main content
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if main is None:
        main = soup

    text = main.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    source: str = "",
    source_type: str = "",
) -> list[Chunk]:
    """Split text into overlapping chunks.

    Tries to split on paragraph boundaries first, then falls back to sentence/character splitting.
    """
    if not text.strip():
        return []

    # First, split by paragraphs
    paragraphs = re.split(r"\n{2,}", text.strip())

    chunks: list[Chunk] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds chunk_size, finalize current and start new
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(Chunk(
                content=current.strip(),
                index=len(chunks),
                source=source,
                source_type=source_type,
            ))
            # Overlap: keep the tail of the previous chunk
            if chunk_overlap > 0 and len(current) > chunk_overlap:
                current = current[-chunk_overlap:] + "\n\n" + para
            else:
                current = para
        else:
            current = current + "\n\n" + para if current else para

        # If a single paragraph exceeds chunk_size, split it by sentences
        while len(current) > chunk_size:
            split_point = _find_split_point(current, chunk_size)
            chunks.append(Chunk(
                content=current[:split_point].strip(),
                index=len(chunks),
                source=source,
                source_type=source_type,
            ))
            if chunk_overlap > 0 and split_point > chunk_overlap:
                current = current[split_point - chunk_overlap:]
            else:
                current = current[split_point:]

    if current.strip():
        chunks.append(Chunk(
            content=current.strip(),
            index=len(chunks),
            source=source,
            source_type=source_type,
        ))

    return chunks


def _find_split_point(text: str, max_len: int) -> int:
    """Find the best split point near max_len, preferring sentence boundaries."""
    # Try to split at sentence boundary
    for pattern in [r"[.!?。！？]\s+", r"[;；]\s+", r"\n"]:
        matches = list(re.finditer(pattern, text[:max_len + 50]))
        if matches:
            best = matches[-1].end()
            if best >= max_len * 0.5:
                return best
    # Fallback: split at space
    space = text.rfind(" ", max_len * 0.5, max_len)
    if space > 0:
        return space
    return max_len


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex boundaries."""
    # Split on sentence-ending punctuation, semicolons, and newlines
    parts = re.split(r"([.!?。！？]\s+|[;；]\s+|\n+)", text)

    sentences: list[str] = []
    current = ""
    for part in parts:
        current += part
        # If this part is a delimiter, finalize the sentence
        if re.fullmatch(r"[.!?。！？]\s+|[;；]\s+|\n+", part):
            stripped = current.strip()
            if stripped:
                sentences.append(stripped)
            current = ""

    if current.strip():
        sentences.append(current.strip())

    return sentences


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_text_semantic(
    text: str,
    embeddings,
    similarity_threshold: float = 0.5,
    min_chunk_size: int = 200,
    max_chunk_size: int = 1500,
    chunk_overlap: int = 100,
    source: str = "",
    source_type: str = "",
) -> list[Chunk]:
    """Split text into chunks based on embedding similarity between sentences.

    Computes cosine similarity between adjacent sentence embeddings and
    breaks at semantic boundaries (low similarity points).
    """
    if not text.strip():
        return []

    sentences = _split_into_sentences(text)

    # Fall back to fixed-size chunking for very short texts
    if len(sentences) < 3:
        return chunk_text(text, chunk_size=max_chunk_size, chunk_overlap=chunk_overlap,
                          source=source, source_type=source_type)

    # Embed all sentences (embed_documents handles batching internally)
    vectors = embeddings.embed_documents(sentences)

    # Compute similarity between adjacent pairs
    similarities = []
    for i in range(len(vectors) - 1):
        similarities.append(_cosine_similarity(vectors[i], vectors[i + 1]))

    # Identify breakpoints where similarity drops below threshold
    breakpoints = [0]
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            breakpoints.append(i + 1)
    breakpoints.append(len(sentences))

    # Build raw segments from breakpoints
    raw_segments: list[list[str]] = []
    for i in range(len(breakpoints) - 1):
        seg = sentences[breakpoints[i]:breakpoints[i + 1]]
        if seg:
            raw_segments.append(seg)

    if not raw_segments:
        return chunk_text(text, chunk_size=max_chunk_size, chunk_overlap=chunk_overlap,
                          source=source, source_type=source_type)

    # Enforce size constraints: merge small, split large
    merged_segments: list[list[str]] = []
    for seg in raw_segments:
        seg_text = " ".join(seg)
        if merged_segments and len(seg_text) < min_chunk_size:
            # Merge with previous segment
            merged_segments[-1].extend(seg)
        elif len(seg_text) > max_chunk_size:
            # Split oversized segment using fixed-size chunking
            merged_segments.append(seg)  # keep as-is, will be split later
        else:
            merged_segments.append(seg)

    # Build chunks
    chunks: list[Chunk] = []
    for seg in merged_segments:
        seg_text = " ".join(seg)
        if len(seg_text) > max_chunk_size:
            # Oversized: fall back to fixed-size splitting within this segment
            sub_chunks = chunk_text(seg_text, chunk_size=max_chunk_size,
                                    chunk_overlap=chunk_overlap, source=source,
                                    source_type=source_type)
            # Re-index
            for c in sub_chunks:
                c.index = len(chunks)
                chunks.append(c)
        else:
            chunks.append(Chunk(
                content=seg_text,
                index=len(chunks),
                source=source,
                source_type=source_type,
            ))

    # Add overlap between consecutive chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        for i in range(1, len(chunks)):
            prev = chunks[i - 1].content
            if len(prev) > chunk_overlap:
                overlap_text = prev[-chunk_overlap:]
                chunks[i].content = overlap_text + "\n\n" + chunks[i].content

    return chunks


async def ingest_text(
    content: str,
    source_name: str,
    source_type: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    strategy: str = "fixed",
    similarity_threshold: float = 0.5,
    min_chunk_size: int = 200,
    max_chunk_size: int = 1500,
) -> list[Chunk]:
    """Ingest plain text: chunk and return."""
    if strategy == "semantic":
        from agent.storage import get_embeddings
        return chunk_text_semantic(
            text=content,
            embeddings=get_embeddings(),
            similarity_threshold=similarity_threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            source=source_name,
            source_type=source_type,
        )
    return chunk_text(
        text=content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source=source_name,
        source_type=source_type,
    )


async def ingest_pdf(
    data: bytes,
    filename: str = "document.pdf",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    strategy: str = "fixed",
    similarity_threshold: float = 0.5,
    min_chunk_size: int = 200,
    max_chunk_size: int = 1500,
) -> list[Chunk]:
    """Ingest a PDF: extract text → chunk."""
    text = extract_text_from_pdf(data)
    if not text.strip():
        return []
    if strategy == "semantic":
        from agent.storage import get_embeddings
        return chunk_text_semantic(
            text=text,
            embeddings=get_embeddings(),
            similarity_threshold=similarity_threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            source=filename,
            source_type="pdf",
        )
    return chunk_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source=filename,
        source_type="pdf",
    )


async def ingest_url(
    url: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    strategy: str = "fixed",
    similarity_threshold: float = 0.5,
    min_chunk_size: int = 200,
    max_chunk_size: int = 1500,
) -> tuple[str, list[Chunk]]:
    """Ingest a URL: fetch → extract → chunk.

    Returns (title, chunks).
    """
    title, text = extract_text_from_url(url)
    if not text.strip():
        return title, []
    if strategy == "semantic":
        from agent.storage import get_embeddings
        chunks = chunk_text_semantic(
            text=text,
            embeddings=get_embeddings(),
            similarity_threshold=similarity_threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            source=url,
            source_type="url",
        )
    else:
        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=url,
            source_type="url",
        )
    return title, chunks
