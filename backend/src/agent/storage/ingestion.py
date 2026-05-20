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


async def ingest_text(
    content: str,
    source_name: str,
    source_type: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Ingest plain text: chunk and return."""
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
) -> list[Chunk]:
    """Ingest a PDF: extract text → chunk."""
    text = extract_text_from_pdf(data)
    if not text.strip():
        return []
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
) -> tuple[str, list[Chunk]]:
    """Ingest a URL: fetch → extract → chunk.

    Returns (title, chunks).
    """
    title, text = extract_text_from_url(url)
    if not text.strip():
        return title, []
    chunks = chunk_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source=url,
        source_type="url",
    )
    return title, chunks
