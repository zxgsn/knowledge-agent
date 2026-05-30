"""Tests for thread_id isolation in recall memory."""

from unittest.mock import MagicMock, patch

import pytest


class TestSearchRecallThreadId:
    """Test search_recall thread_id filtering."""

    @patch("agent.storage.get_embeddings")
    @patch("agent.db.recall.get_conn")
    def test_search_recall_with_thread_id(self, mock_conn, mock_embeddings):
        """search_recall passes thread_id to SQL WHERE clause when provided."""
        from agent.db import search_recall

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1] * 10
        mock_embeddings.return_value = mock_emb

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        search_recall("test query", limit=5, thread_id="session-abc")

        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "WHERE thread_id = %s" in sql
        assert "session-abc" in params

    @patch("agent.storage.get_embeddings")
    @patch("agent.db.recall.get_conn")
    def test_search_recall_without_thread_id(self, mock_conn, mock_embeddings):
        """search_recall without thread_id has no thread_id WHERE clause."""
        from agent.db import search_recall

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1] * 10
        mock_embeddings.return_value = mock_emb

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        search_recall("test query", limit=5)

        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]
        assert "WHERE thread_id" not in sql


class TestGetRecentRecallThreadId:
    """Test get_recent_recall thread_id filtering."""

    @patch("agent.storage.ensure_recall_table")
    @patch("agent.db.recall.get_conn")
    def test_get_recent_recall_with_thread_id(self, mock_conn, mock_ensure):
        """get_recent_recall filters by thread_id when provided."""
        from agent.db import get_recent_recall

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        get_recent_recall(limit=3, thread_id="session-xyz")

        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "WHERE thread_id = %s" in sql
        assert "session-xyz" in params

    @patch("agent.storage.ensure_recall_table")
    @patch("agent.db.recall.get_conn")
    def test_get_recent_recall_without_thread_id(self, mock_conn, mock_ensure):
        """get_recent_recall without thread_id returns all threads."""
        from agent.db import get_recent_recall

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        get_recent_recall(limit=3)

        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]
        assert "WHERE" not in sql


class TestSaveToRecallThreadId:
    """Test save_to_recall passes thread_id correctly."""

    @patch("agent.storage.ensure_recall_table")
    @patch("agent.storage.get_embeddings")
    @patch("agent.db.recall.get_conn")
    def test_save_to_recall_custom_thread(self, mock_conn, mock_embeddings, mock_ensure):
        """save_to_recall stores with custom thread_id."""
        from agent.db import save_to_recall

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1] * 10
        mock_embeddings.return_value = mock_emb

        mock_cursor = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        save_to_recall("user", "hello", thread_id="my-session")

        call_args = mock_cursor.execute.call_args
        params = call_args[0][1]
        assert "my-session" in params
