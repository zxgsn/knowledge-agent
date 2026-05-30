"""Tests for agent.db — centralized DB helpers with mocked connections."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_conn():
    """Create a mock psycopg connection."""
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    return conn


def _make_mock_embeddings():
    """Create a mock DashScopeEmbeddings."""
    emb = MagicMock()
    emb.embed_query.return_value = [0.1] * 1024
    emb.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024]
    return emb


@contextmanager
def _mock_conn_ctx(conn):
    """Context manager that yields the mock conn."""
    yield conn


class TestSearchArchival:
    @patch("agent.db.archival.get_conn")
    @patch("agent.storage.get_embeddings")
    def test_returns_results(self, mock_get_emb, mock_get_conn):
        from agent.db import search_archival

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)
        mock_get_emb.return_value = _make_mock_embeddings()

        # Mock DB rows: (content, metadata, namespace, score)
        mock_row = ("test content", json.dumps({"source": "test"}), "research", 0.85)
        conn.execute.return_value.fetchall.return_value = [mock_row]

        with patch("agent.storage.reranker.rerank", side_effect=lambda q, r, top_k: r[:top_k]):
            results = search_archival("test query", limit=5)

        assert len(results) == 1
        assert results[0]["content"] == "test content"
        assert results[0]["score"] == 0.85
        assert results[0]["metadata"] == {"source": "test"}

    @patch("agent.db.archival.get_conn")
    @patch("agent.storage.get_embeddings")
    def test_filters_low_scores(self, mock_get_emb, mock_get_conn):
        from agent.db import search_archival

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)
        mock_get_emb.return_value = _make_mock_embeddings()

        mock_row = ("low score content", json.dumps({}), "ns", 0.005)
        conn.execute.return_value.fetchall.return_value = [mock_row]

        results = search_archival("test query")
        assert len(results) == 0

    @patch("agent.db.archival.get_conn")
    @patch("agent.storage.get_embeddings")
    def test_embedding_failure_returns_empty(self, mock_get_emb, mock_get_conn):
        from agent.db import search_archival

        mock_get_emb.side_effect = RuntimeError("API error")
        assert search_archival("test") == []

    @patch("agent.db.archival.get_conn")
    @patch("agent.storage.get_embeddings")
    def test_db_failure_returns_empty(self, mock_get_emb, mock_get_conn):
        from agent.db import search_archival

        mock_get_emb.return_value = _make_mock_embeddings()
        mock_get_conn.side_effect = RuntimeError("connection failed")
        assert search_archival("test") == []


class TestSearchArchivalForDedup:
    @patch("agent.db.archival.get_conn")
    @patch("agent.storage.get_embeddings")
    def test_returns_results_with_id(self, mock_get_emb, mock_get_conn):
        from agent.db import search_archival_for_dedup

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)
        mock_get_emb.return_value = _make_mock_embeddings()

        mock_row = ("id-1", "fact content", json.dumps({}), 0.9)
        conn.execute.return_value.fetchall.return_value = [mock_row]

        results = search_archival_for_dedup("query", namespace="conversation_facts")
        assert len(results) == 1
        assert results[0]["id"] == "id-1"
        assert results[0]["content"] == "fact content"

    @patch("agent.db.archival.get_conn")
    @patch("agent.storage.get_embeddings")
    def test_embedding_failure_returns_empty(self, mock_get_emb, mock_get_conn):
        from agent.db import search_archival_for_dedup

        mock_get_emb.side_effect = RuntimeError("fail")
        assert search_archival_for_dedup("q") == []


class TestPutToArchival:
    @patch("agent.db.archival.get_conn")
    @patch("agent.storage.get_embeddings")
    def test_returns_entry_id(self, mock_get_emb, mock_get_conn):
        from agent.db import put_to_archival

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)
        mock_get_emb.return_value = _make_mock_embeddings()

        entry_id = put_to_archival("content", "test_ns", {"key": "val"})
        assert isinstance(entry_id, str)
        assert len(entry_id) == 36  # UUID format
        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

    @patch("agent.db.archival.get_conn")
    @patch("agent.storage.get_embeddings")
    def test_with_document_id(self, mock_get_emb, mock_get_conn):
        from agent.db import put_to_archival

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)
        mock_get_emb.return_value = _make_mock_embeddings()

        entry_id = put_to_archival("content", "ns", {}, document_id="doc-123")
        assert isinstance(entry_id, str)
        # Verify document_id was passed in the SQL args
        call_args = conn.execute.call_args
        assert "doc-123" in call_args[0][1]

    @patch("agent.db.archival.get_conn")
    @patch("agent.storage.get_embeddings")
    def test_on_conflict_update(self, mock_get_emb, mock_get_conn):
        from agent.db import put_to_archival

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)
        mock_get_emb.return_value = _make_mock_embeddings()

        put_to_archival("content", "ns", {}, on_conflict="update")
        sql = conn.execute.call_args[0][0]
        assert "DO UPDATE" in sql


class TestInsertDocument:
    @patch("agent.db.documents._ensure_content_hash_column")
    @patch("agent.db.documents.get_conn")
    def test_new_document(self, mock_get_conn, mock_ensure_hash):
        from agent.db import insert_document

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)

        # Simulate RETURNING id succeeding (new document)
        mock_result = MagicMock()
        mock_result.fetchone.return_value = ("doc-id-123",)
        conn.execute.return_value = mock_result

        doc_id, is_new = insert_document("title", "source", "url", "full text", 5)
        assert doc_id == "doc-id-123"
        assert is_new is True

    @patch("agent.db.documents._ensure_content_hash_column")
    @patch("agent.db.documents.get_conn")
    def test_existing_document(self, mock_get_conn, mock_ensure_hash):
        from agent.db import insert_document

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)

        # First call (INSERT) returns None (conflict)
        # Second call (SELECT) returns existing id
        mock_insert_result = MagicMock()
        mock_insert_result.fetchone.return_value = None
        mock_select_result = MagicMock()
        mock_select_result.fetchone.return_value = ("existing-id",)
        conn.execute.side_effect = [mock_insert_result, mock_select_result]

        doc_id, is_new = insert_document("title", "source", "url", "text", 1)
        assert doc_id == "existing-id"
        assert is_new is False


class TestSaveToRecall:
    @patch("agent.db.recall.get_conn")
    @patch("agent.storage.get_embeddings")
    @patch("agent.storage.ensure_recall_table")
    def test_saves_message(self, mock_ensure, mock_get_emb, mock_get_conn):
        from agent.db import save_to_recall

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)
        mock_get_emb.return_value = _make_mock_embeddings()

        save_to_recall("user", "hello world", "thread-1")
        mock_ensure.assert_called_once()
        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

        # Verify the SQL args contain role and content
        call_args = conn.execute.call_args[0]
        sql_params = call_args[1]
        assert "user" in sql_params
        assert "hello world" in sql_params
        assert "thread-1" in sql_params


class TestSearchRecall:
    @patch("agent.db.recall.get_conn")
    @patch("agent.storage.get_embeddings")
    def test_returns_results(self, mock_get_emb, mock_get_conn):
        from agent.db import search_recall

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)
        mock_get_emb.return_value = _make_mock_embeddings()

        # Mock row: (id, thread_id, role, content, metadata, score)
        mock_row = ("r1", "t1", "user", "hello", json.dumps({}), 0.9)
        conn.execute.return_value.fetchall.return_value = [mock_row]

        with patch("agent.storage.reranker.rerank", side_effect=lambda q, r, top_k: r[:top_k]):
            results = search_recall("query", limit=5)

        assert len(results) == 1
        assert results[0]["id"] == "r1"
        assert results[0]["role"] == "user"

    @patch("agent.db.recall.get_conn")
    @patch("agent.storage.get_embeddings")
    def test_embedding_failure_returns_empty(self, mock_get_emb, mock_get_conn):
        from agent.db import search_recall

        mock_get_emb.side_effect = RuntimeError("fail")
        assert search_recall("q") == []


class TestUpdateArchival:
    @patch("agent.db.archival.get_conn")
    @patch("agent.storage.get_embeddings")
    def test_executes_update(self, mock_get_emb, mock_get_conn):
        from agent.db import update_archival

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)
        mock_get_emb.return_value = _make_mock_embeddings()

        # _snapshot_version does 3 execute calls: SELECT content, SELECT MAX(version), INSERT snapshot
        mock_select = MagicMock()
        mock_select.fetchone.return_value = ("old content", "{}", "[0.1]")
        mock_max_ver = MagicMock()
        mock_max_ver.fetchone.return_value = (0,)
        conn.execute.side_effect = [mock_select, mock_max_ver, MagicMock(), MagicMock()]

        update_archival("entry-id", "new content", {"source": "test"})
        assert conn.execute.call_count == 4  # 3 snapshot + 1 update
        conn.commit.assert_called_once()

        # Last execute call is the actual UPDATE
        sql = conn.execute.call_args_list[-1][0][0]
        assert "UPDATE" in sql


class TestDeleteArchival:
    @patch("agent.db.archival.get_conn")
    def test_returns_true_on_delete(self, mock_get_conn):
        from agent.db import delete_archival

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)

        # _snapshot_version: SELECT, SELECT MAX, INSERT; then soft-delete UPDATE
        mock_select = MagicMock()
        mock_select.fetchone.return_value = ("content", "{}", "[0.1]")
        mock_max_ver = MagicMock()
        mock_max_ver.fetchone.return_value = (0,)
        mock_update = MagicMock()
        mock_update.rowcount = 1
        conn.execute.side_effect = [mock_select, mock_max_ver, MagicMock(), mock_update]

        assert delete_archival("entry-id") is True

    @patch("agent.db.archival.get_conn")
    def test_returns_false_when_no_match(self, mock_get_conn):
        from agent.db import delete_archival

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)

        mock_select = MagicMock()
        mock_select.fetchone.return_value = ("content", "{}", "[0.1]")
        mock_max_ver = MagicMock()
        mock_max_ver.fetchone.return_value = (0,)
        mock_update = MagicMock()
        mock_update.rowcount = 0
        conn.execute.side_effect = [mock_select, mock_max_ver, MagicMock(), mock_update]

        assert delete_archival("nonexistent") is False


class TestGetAllFacts:
    @patch("agent.db.archival.get_conn")
    def test_returns_facts(self, mock_get_conn):
        from agent.db import get_all_facts

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)
        mock_row = ("id-1", "fact text", json.dumps({}), "[0.1, 0.2]")
        conn.execute.return_value.fetchall.return_value = [mock_row]

        results = get_all_facts("conversation_facts", 100)
        assert len(results) == 1
        assert results[0]["id"] == "id-1"
        assert results[0]["content"] == "fact text"
        assert results[0]["embedding_text"] == "[0.1, 0.2]"

    @patch("agent.db.archival.get_conn")
    def test_db_failure_returns_empty(self, mock_get_conn):
        from agent.db import get_all_facts

        mock_get_conn.side_effect = RuntimeError("fail")
        assert get_all_facts() == []


class TestCleanupNamespace:
    @patch("agent.db.archival.get_conn")
    def test_returns_deleted_count(self, mock_get_conn):
        from agent.db import cleanup_namespace

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)
        conn.execute.return_value.rowcount = 5

        assert cleanup_namespace("ingested", 30) == 5

    @patch("agent.db.archival.get_conn")
    def test_failure_returns_zero(self, mock_get_conn):
        from agent.db import cleanup_namespace

        mock_get_conn.side_effect = RuntimeError("fail")
        assert cleanup_namespace("ingested", 30) == 0


class TestCleanupExcess:
    @patch("agent.db.archival.get_conn")
    def test_under_limit_returns_zero(self, mock_get_conn):
        from agent.db import cleanup_excess

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)

        mock_count = MagicMock()
        mock_count.fetchone.return_value = (50,)
        conn.execute.return_value = mock_count

        assert cleanup_excess("ns", 100) == 0

    @patch("agent.db.archival.get_conn")
    def test_over_limit_deletes_excess(self, mock_get_conn):
        from agent.db import cleanup_excess

        conn = _make_mock_conn()
        mock_get_conn.return_value = _mock_conn_ctx(conn)

        mock_count = MagicMock()
        mock_count.fetchone.return_value = (150,)

        # SELECT id ... LIMIT 50 returns 50 rows
        mock_select_ids = MagicMock()
        mock_select_ids.fetchall.return_value = [(f"eid-{i}",) for i in range(50)]

        # Each excess entry: snapshot (SELECT, SELECT MAX, INSERT) + soft-delete UPDATE
        calls = [mock_count, mock_select_ids]
        for _ in range(50):
            mock_snap = MagicMock()
            mock_snap.fetchone.return_value = ("content", "{}", "[0.1]")
            mock_max_ver = MagicMock()
            mock_max_ver.fetchone.return_value = (0,)
            calls.extend([mock_snap, mock_max_ver, MagicMock(), MagicMock()])

        conn.execute.side_effect = calls

        result = cleanup_excess("ns", 100)
        assert result == 50

    @patch("agent.db.archival.get_conn")
    def test_failure_returns_zero(self, mock_get_conn):
        from agent.db import cleanup_excess

        mock_get_conn.side_effect = RuntimeError("fail")
        assert cleanup_excess("ns", 100) == 0


class TestDeduplicateResults:
    def test_empty_list(self):
        from agent.db import _deduplicate_results

        assert _deduplicate_results([]) == []

    def test_single_item(self):
        from agent.db import _deduplicate_results

        results = [{"content": "test", "score": 0.9}]
        assert _deduplicate_results(results) == results

    def test_no_duplicates(self):
        from agent.db import _deduplicate_results

        results = [
            {"content": "completely different text", "score": 0.9},
            {"content": "another unique content here", "score": 0.8},
        ]
        assert len(_deduplicate_results(results)) == 2

    def test_removes_duplicate(self):
        from agent.db import _deduplicate_results

        results = [
            {"content": "the quick brown fox jumps over the lazy dog", "score": 0.9},
            {"content": "the quick brown fox jumps over the lazy dog", "score": 0.8},
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 1
        assert deduped[0]["score"] == 0.9  # Keeps higher score

    def test_keeps_similar_but_different(self):
        from agent.db import _deduplicate_results

        results = [
            {"content": "the quick brown fox", "score": 0.9},
            {"content": "a slow red turtle", "score": 0.8},
        ]
        assert len(_deduplicate_results(results)) == 2

    def test_threshold_applied(self):
        from agent.db import _deduplicate_results

        # 90% overlap
        results = [
            {"content": "a b c d e f g h i j", "score": 0.9},
            {"content": "a b c d e f g h i k", "score": 0.8},
        ]
        # With default threshold 0.95, should keep both
        assert len(_deduplicate_results(results, threshold=0.95)) == 2

        # With lower threshold, should dedup
        assert len(_deduplicate_results(results, threshold=0.8)) == 1
