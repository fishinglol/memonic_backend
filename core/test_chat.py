"""
test_chat.py — Tests for Memonic chat endpoints + Ollama helper

Uses a separate SQLite test DB so tests don't touch the real DB.
Mocks call_ollama so no Ollama/GPU is needed.

Run:
    cd ~/memonic_backend/core && python -m pytest test_chat.py -v
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sqlalchemy

# ── Override database BEFORE importing main ──────────────────────────────────
# We override database.py's engine/SessionLocal to use a test DB

TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "test_chat.db")
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_PATH}"


def _setup_test_db():
    """Create a fresh test engine and override the database module."""
    import database

    test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    test_session = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Override the database module globals
    database.engine = test_engine
    database.SessionLocal = test_session

    return test_engine, test_session


_test_engine, _TestSessionLocal = _setup_test_db()

# NOW import main (it will use the overridden database)
import models
import main

# Create tables
models.Base.metadata.create_all(bind=_test_engine)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_db():
    """Clear all rows before each test."""
    db = _TestSessionLocal()
    try:
        db.query(models.ChatMessage).delete()
        db.query(models.User).delete()
        db.commit()
    finally:
        db.close()
    yield
    # Cleanup after test
    db = _TestSessionLocal()
    try:
        db.query(models.ChatMessage).delete()
        db.query(models.User).delete()
        db.commit()
    finally:
        db.close()


@pytest.fixture()
def client():
    """FastAPI TestClient with overridden DB dependency."""
    from fastapi.testclient import TestClient

    def override_get_db():
        db = _TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[main.get_db] = override_get_db
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()


MOCK_AI_REPLY = "Hello! I'm Memonic, how can I help you today?"


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/chat
# ══════════════════════════════════════════════════════════════════════════════

class TestChatEndpoint:

    @patch("main.call_ollama", return_value=MOCK_AI_REPLY)
    def test_chat_new_session(self, mock_ollama, client):
        """POST with no session_id → generates UUID, returns reply + title."""
        res = client.post("/api/chat", json={"message": "Hello world"})
        assert res.status_code == 200

        data = res.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0
        assert data["reply"] == MOCK_AI_REPLY
        assert data["title"] == "Hello world"
        mock_ollama.assert_called_once()

    @patch("main.call_ollama", return_value=MOCK_AI_REPLY)
    def test_chat_continue_session(self, mock_ollama, client):
        """POST with existing session_id → appends to same session."""
        # First message
        res1 = client.post("/api/chat", json={"message": "First message"})
        sid = res1.json()["session_id"]

        # Second message with same session
        res2 = client.post("/api/chat", json={
            "session_id": sid,
            "message": "Second message"
        })
        assert res2.status_code == 200
        assert res2.json()["session_id"] == sid

        # Verify Ollama received conversation history
        # 2nd call should have 3 messages: user1, ai1, user2
        second_call_messages = mock_ollama.call_args_list[1][0][0]
        assert len(second_call_messages) == 3
        assert second_call_messages[0]["role"] == "user"
        assert second_call_messages[0]["content"] == "First message"

    @patch("main.call_ollama", return_value=MOCK_AI_REPLY)
    def test_chat_long_title_truncated(self, mock_ollama, client):
        """Message >50 chars → title gets '...' suffix."""
        long_msg = "A" * 80
        res = client.post("/api/chat", json={"message": long_msg})
        assert res.status_code == 200

        title = res.json()["title"]
        assert len(title) == 53  # 50 chars + "..."
        assert title.endswith("...")

    @patch("main.call_ollama", return_value=MOCK_AI_REPLY)
    def test_chat_messages_saved_to_db(self, mock_ollama, client):
        """Both user and AI messages are persisted to DB."""
        res = client.post("/api/chat", json={"message": "Test message"})
        sid = res.json()["session_id"]

        # Check via the session messages endpoint
        msgs_res = client.get(f"/api/chat/sessions/{sid}")
        assert msgs_res.status_code == 200

        msgs = msgs_res.json()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Test message"
        assert msgs[1]["role"] == "ai"
        assert msgs[1]["content"] == MOCK_AI_REPLY


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/chat/sessions
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionsList:

    def test_sessions_empty(self, client):
        """GET sessions with no data → empty list."""
        res = client.get("/api/chat/sessions")
        assert res.status_code == 200
        assert res.json() == []

    @patch("main.call_ollama", return_value=MOCK_AI_REPLY)
    def test_sessions_list(self, mock_ollama, client):
        """Sessions listed with correct titles after creating chats."""
        client.post("/api/chat", json={"message": "Talk about Python"})
        client.post("/api/chat", json={"message": "Talk about Rust"})

        res = client.get("/api/chat/sessions")
        assert res.status_code == 200

        sessions = res.json()
        assert len(sessions) == 2

        titles = [s["title"] for s in sessions]
        assert "Talk about Python" in titles
        assert "Talk about Rust" in titles

        # Each session has required fields
        for s in sessions:
            assert "session_id" in s
            assert "title" in s
            assert "created_at" in s


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/chat/sessions/{session_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionMessages:

    @patch("main.call_ollama", return_value=MOCK_AI_REPLY)
    def test_session_messages(self, mock_ollama, client):
        """GET messages for a session → returns in order."""
        res = client.post("/api/chat", json={"message": "Hello"})
        sid = res.json()["session_id"]

        msgs_res = client.get(f"/api/chat/sessions/{sid}")
        assert msgs_res.status_code == 200

        msgs = msgs_res.json()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "ai"
        for m in msgs:
            assert "timestamp" in m

    def test_session_not_found(self, client):
        """GET non-existent session → 404."""
        res = client.get("/api/chat/sessions/nonexistent-uuid-12345")
        assert res.status_code == 404
        assert res.json()["detail"] == "Session not found"


# ══════════════════════════════════════════════════════════════════════════════
# call_ollama helper
# ══════════════════════════════════════════════════════════════════════════════

class TestCallOllama:

    @patch("main.requests.post")
    def test_ollama_success(self, mock_post):
        """call_ollama parses a valid Ollama response correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "I am Memonic AI."}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = main.call_ollama([{"role": "user", "content": "Who are you?"}])
        assert result == "I am Memonic AI."

        # Verify correct model was used
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["model"] == "llama3.2:3b"
        assert call_kwargs[1]["json"]["stream"] is False

    def test_ollama_connection_error(self):
        """call_ollama returns warning when Ollama is down."""
        import requests as req

        with patch("main.requests.post", side_effect=req.exceptions.ConnectionError("refused")):
            result = main.call_ollama([{"role": "user", "content": "Hello"}])
            assert "Cannot connect to Ollama" in result
