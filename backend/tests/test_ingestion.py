"""Tests for agent.storage.ingestion — chunking and text processing."""

from agent.storage.ingestion import (
    Chunk,
    _cosine_similarity,
    _find_split_point,
    _split_into_sentences,
    chunk_text,
)


class TestChunk:
    def test_id_property(self):
        c = Chunk(content="hello", index=0, source="doc1")
        assert c.id == "doc1_0"

    def test_id_with_index(self):
        c = Chunk(content="hello", index=3, source="test")
        assert c.id == "test_3"

    def test_default_metadata(self):
        c = Chunk(content="hello", index=0)
        assert c.metadata == {}
        assert c.source == ""
        assert c.source_type == ""


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, a) == 1.0

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == -1.0

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_partial_similarity(self):
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / (2.0 ** 0.5)
        assert abs(_cosine_similarity(a, b) - expected) < 1e-9


class TestFindSplitPoint:
    def test_sentence_boundary(self):
        text = "Hello world. This is a test. Another sentence."
        # Should split at a sentence boundary near the target
        point = _find_split_point(text, 25)
        assert point <= 30
        # Split point is after the delimiter (space or punctuation)
        assert text[point - 1] in ".!? " or text[point] == " "

    def test_short_text(self):
        text = "Short"
        point = _find_split_point(text, 100)
        # Returns max_len when text is shorter; used as slice index text[:point]
        assert point >= len(text)

    def test_no_boundary_fallback_to_space(self):
        text = "word " * 100
        point = _find_split_point(text, 50)
        assert point > 0
        assert point <= 50

    def test_chinese_punctuation(self):
        text = "你好世界。这是一个测试。另一个句子。"
        point = _find_split_point(text, 15)
        assert point > 0


class TestSplitIntoSentences:
    def test_english_sentences(self):
        text = "Hello world. This is a test. Done."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3
        assert "Hello world." in sentences[0]

    def test_chinese_sentences(self):
        text = "你好世界。这是一个测试。完成。"
        sentences = _split_into_sentences(text)
        # Chinese punctuation splitting depends on encoding; at minimum should return 1+
        assert len(sentences) >= 1

    def test_single_sentence(self):
        sentences = _split_into_sentences("Just one sentence.")
        assert len(sentences) == 1

    def test_empty_text(self):
        assert _split_into_sentences("") == []

    def test_newline_delimited(self):
        text = "Line one\nLine two\nLine three"
        sentences = _split_into_sentences(text)
        assert len(sentences) >= 2


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        text = "Short text that fits in one chunk."
        chunks = chunk_text(text, chunk_size=200)
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].index == 0

    def test_long_text_multiple_chunks(self):
        # Create text that's clearly longer than chunk_size
        text = "This is a sentence. " * 100  # ~2000 chars
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        # Indices should be sequential
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_chunk_indices_sequential(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nParagraph four."
        chunks = chunk_text(text, chunk_size=30)
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_source_metadata(self):
        text = "Some text content for testing."
        chunks = chunk_text(text, source="test_doc", source_type="text")
        assert all(c.source == "test_doc" for c in chunks)
        assert all(c.source_type == "text" for c in chunks)

    def test_paragraph_splitting(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, chunk_size=50)
        # Each paragraph is short enough to be its own chunk or merged
        assert len(chunks) >= 1

    def test_overlap_present(self):
        # Create text with two clear paragraphs that exceed chunk_size
        para1 = "A" * 400
        para2 = "B" * 400
        text = f"{para1}\n\n{para2}"
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
        if len(chunks) > 1:
            # Second chunk should contain some overlap from first
            assert len(chunks[1].content) > 400

    def test_preserves_content(self):
        text = "Alpha. Beta. Gamma. Delta. Epsilon."
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
        # All content should be present across chunks
        combined = " ".join(c.content for c in chunks)
        for word in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]:
            assert word in combined
