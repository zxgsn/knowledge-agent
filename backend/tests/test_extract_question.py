"""Tests for _extract_question helper in ingest flow."""

import pytest

from agent.nodes.memory_manager import _extract_question


class TestExtractQuestion:
    """Test _extract_question for different source types."""

    def test_pdf_with_question(self):
        content = "请帮我总结这个文档 [UPLOAD_PDF:report.pdf]base64data[/UPLOAD_PDF]"
        result = _extract_question(content, "pdf", "base64data")
        assert result == "请帮我总结这个文档"

    def test_pdf_no_question(self):
        content = "[UPLOAD_PDF:report.pdf]base64data[/UPLOAD_PDF]"
        result = _extract_question(content, "pdf", "base64data")
        assert result == ""

    def test_pdf_whitespace_only_question(self):
        content = "   [UPLOAD_PDF:report.pdf]base64data[/UPLOAD_PDF]"
        result = _extract_question(content, "pdf", "base64data")
        assert result == ""

    def test_url_with_question(self):
        content = "帮我分析一下这个网页 https://example.com/article"
        result = _extract_question(content, "url", "https://example.com/article")
        assert result == "帮我分析一下这个网页"

    def test_url_no_question(self):
        content = "https://example.com/article"
        result = _extract_question(content, "url", "https://example.com/article")
        assert result == ""

    def test_text_no_question(self):
        result = _extract_question("some text content", "text", "some text content")
        assert result == ""
