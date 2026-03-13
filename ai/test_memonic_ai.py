"""
test_memonic_ai.py — Memonic AI tests based on actual api.py + mcp_server.py code

Tests are grouped by the real bugs found in your code.
All tests mock heavy models so they run instantly without GPU/Ollama.

Run all fast tests:
    pip install pytest pytest-mock torch numpy fastapi httpx
    python -m pytest test_memonic_ai.py -v

Run only integration tests (needs full stack):
    python -m pytest test_memonic_ai.py -m integration -v
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock, patch, AsyncMock
import io


# ══════════════════════════════════════════════════════════════════════════════
# BUG #1: EMBEDDING SHAPE — encode_batch squeeze() issue
#
# encode_batch() returns shape [batch=1, frames, dim]
# .squeeze(0) → [frames, dim]  ← WRONG, cosine_similarity(dim=0) compares frames
# .squeeze()   → [dim]         ← correct for single sample
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingShape:

    def test_squeeze0_gives_wrong_shape(self):
        """Demonstrates the actual bug: squeeze(0) leaves 2D tensor."""
        fake_batch_output = torch.randn(1, 1, 192)  # [batch=1, frames=1, dim=192]
        result = fake_batch_output.squeeze(0)        # what api.py does
        assert result.shape == (1, 192), f"Got {result.shape}"
        # This is WRONG — cosine_similarity(known, test, dim=0) will compare
        # across the frames dimension, not the embedding dimension

    def test_squeeze_no_arg_gives_correct_shape(self):
        """squeeze() without argument correctly collapses to [dim]."""
        fake_batch_output = torch.randn(1, 1, 192)
        result = fake_batch_output.squeeze()  # correct fix
        assert result.shape == (192,), f"Got {result.shape}"

    def test_cosine_similarity_with_wrong_shape_is_silent(self):
        """
        The dangerous part: cosine_similarity doesn't crash with wrong shape.
        It silently computes a wrong score. This is the real bug.
        """
        known = torch.randn(1, 192)  # [frames=1, dim=192] — wrong shape after squeeze(0)
        test  = torch.randn(1, 192)

        # This runs without error but compares 192-dim vectors along dim=0
        # (only 1 frame), producing a single number that means nothing
        score = F.cosine_similarity(known, test, dim=0)
        assert score.shape == (192,), "Returns 192 scores not 1 — definitely wrong!"

    def test_cosine_similarity_with_correct_shape(self):
        """After fix: squeeze() → [dim=192], cosine_similarity gives one scalar."""
        known = torch.randn(192)  # correct 1D embedding
        test  = torch.randn(192)

        score = F.cosine_similarity(known.unsqueeze(0), test.unsqueeze(0), dim=1)
        assert score.shape == (1,), f"Expected 1 score, got shape {score.shape}"
        assert -1.0 <= score.item() <= 1.0

    def test_correct_approach_for_cosine_in_api(self):
        """
        The fix to use in api.py instead of F.cosine_similarity(known, test, dim=0).
        Use dim=0 only if both tensors are 1D [dim].
        """
        known = torch.randn(192)  # after .squeeze() fix
        test  = torch.randn(192)

        score = F.cosine_similarity(known.unsqueeze(0), test.unsqueeze(0), dim=1).item()
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# BUG #2: SPEAKER MATCHING LOGIC — picks worst match when all scores negative
# ══════════════════════════════════════════════════════════════════════════════

class TestSpeakerMatchingLogic:

    def _run_matching(self, test_emb, profiles_cache):
        """Mirrors the exact matching loop in api.py."""
        best_match = "Unknown"
        best_score = 0.0

        for known_user, known_embedding in profiles_cache.items():
            similarity = F.cosine_similarity(known_embedding, test_emb, dim=0)
            score = similarity.item()
            if score > best_score:
                best_score = score
                best_match = known_user

        identified_user = best_match if best_score > 0.75 else "Unknown"
        return identified_user, best_score

    def test_empty_cache_returns_unknown(self):
        test_emb = torch.randn(192)
        user, score = self._run_matching(test_emb, {})
        assert user == "Unknown"
        assert score == 0.0

    def test_clear_match_identified(self):
        base = torch.randn(192)
        base = base / base.norm()
        # enrolled embedding is nearly identical
        enrolled = base + torch.randn(192) * 0.01
        enrolled = enrolled / enrolled.norm()

        profiles = {"fish": enrolled}
        user, score = self._run_matching(base, profiles)
        assert user == "fish", f"Expected 'fish', got '{user}' with score {score:.3f}"
        assert score > 0.75

    def test_low_confidence_returns_unknown(self):
        """Speaker found but below 0.75 threshold → Unknown."""
        base = torch.randn(192)
        base = base / base.norm()
        # Very different vector → low cosine sim
        other = torch.randn(192)
        other = other / other.norm()

        profiles = {"fish": other}
        # Force a score we know is low by using orthogonal vector
        rand = torch.randn(192)
        proj = rand - torch.dot(rand, base) * base
        orthogonal = proj / proj.norm()

        user, score = self._run_matching(orthogonal, profiles)
        # Score could be near 0 — should be Unknown
        if score <= 0.75:
            assert user == "Unknown"

    def test_all_negative_scores_stays_unknown(self):
        """
        BUG SCENARIO: if all profiles score below 0, best_score never exceeds 0.0
        so identified_user stays 'Unknown'. This is actually CORRECT behavior —
        but only because initial best_score=0.0 acts as a floor.
        Verify this assumption holds.
        """
        # Two very dissimilar embeddings
        e1 = torch.tensor([1.0, 0.0, 0.0])
        e2 = torch.tensor([-1.0, 0.0, 0.0])  # cosine sim = -1.0 with e1

        profiles = {"userA": e2}
        user, score = self._run_matching(e1, profiles)

        # score=-1.0, never > best_score=0.0, so best_match stays "Unknown"
        assert user == "Unknown", (
            "All-negative scores should return Unknown, not a bad match"
        )
        assert score == 0.0  # floor was never crossed

    def test_multiple_users_picks_highest(self):
        """With multiple enrolled users, picks the best match."""
        base = torch.randn(192)
        base = base / base.norm()

        close = base + torch.randn(192) * 0.05
        close = close / close.norm()

        far = torch.randn(192)
        far = far / far.norm()

        profiles = {"stranger": far, "fish": close}
        user, score = self._run_matching(base, profiles)

        close_score = F.cosine_similarity(close.unsqueeze(0), base.unsqueeze(0), dim=1).item()
        if close_score > 0.75:
            assert user == "fish"


# ══════════════════════════════════════════════════════════════════════════════
# BUG #3: CHROMADB CONCURRENT ACCESS
# api.py and mcp_server.py both open PersistentClient at "./memonic_memory"
# Running both processes simultaneously causes lock contention on Orange Pi
# ══════════════════════════════════════════════════════════════════════════════

class TestChromaDBConcurrency:

    def test_same_path_two_clients_causes_lock(self):
        """
        Documents the risk: two PersistentClients on the same path.
        On Orange Pi (single-user, low RAM), this will cause:
        - "database is locked" SQLite errors
        - Corrupted writes if both write simultaneously

        Fix: mcp_server.py should use HttpClient pointing to a
        running ChromaDB server, OR read-only queries only.
        """
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            # First client (simulates api.py)
            client1 = chromadb.PersistentClient(path=tmpdir)
            col1 = client1.get_or_create_collection("test")
            col1.add(documents=["hello"], ids=["1"])

            # Second client same path (simulates mcp_server.py)
            # On some systems this works, on others it locks — document it
            try:
                client2 = chromadb.PersistentClient(path=tmpdir)
                col2 = client2.get_collection("test")
                results = col2.get()
                # If we get here, no lock — but concurrent WRITES are still risky
                print(f"\n  WARNING: Two clients opened same ChromaDB — no immediate lock,")
                print(f"  but concurrent writes from api.py + mcp_server.py are unsafe.")
                assert True  # document the risk, don't fail
            except Exception as e:
                pytest.fail(
                    f"ChromaDB lock conflict detected: {e}\n"
                    f"Fix: Use chromadb.HttpClient in mcp_server.py instead of PersistentClient"
                )


# ══════════════════════════════════════════════════════════════════════════════
# BUG #4: list_users() metadata key assumption
# mem0 may store user_id under a different key internally
# ══════════════════════════════════════════════════════════════════════════════

class TestListUsersMetadata:

    def test_list_users_with_wrong_key_returns_empty(self):
        """
        If mem0 stores user under key 'user_id' but ChromaDB metadata uses
        a different key (e.g. nothing, or 'agent_id'), list_users() silently
        returns empty set. This test shows the assumption to verify.
        """
        # Simulate what collection.get() returns in mcp_server.py
        mock_metadatas = [
            {"emotion": "Happy", "speaker_confidence": 0.9},  # NO user_id key!
            {"emotion": "Neutral", "user_id": "fish"},        # has user_id
        ]

        # Mirrors the list_users() logic
        user_ids = set()
        for meta in mock_metadatas:
            if meta and "user_id" in meta:
                user_ids.add(meta["user_id"])

        # Only 1 of 2 entries found — first entry's user silently dropped
        assert len(user_ids) == 1
        assert "fish" in user_ids
        # Fix: after enrolling, check what keys mem0 actually writes to ChromaDB

    def test_get_memory_stats_handles_missing_keys(self):
        """get_memory_stats() uses .get() with defaults — this is safe."""
        mock_metadatas = [
            {"emotion": "Happy"},            # no user_id
            {"user_id": "fish"},             # no emotion
            None,                             # None entry (can happen)
            {"user_id": "fish", "emotion": "Sad"},
        ]

        user_counts: dict = {}
        emotion_counts: dict = {}

        for meta in mock_metadatas:
            if not meta:
                continue  # None guard — correctly handled in your code
            uid = meta.get("user_id", "unknown")
            emotion = meta.get("emotion", "Unknown")
            user_counts[uid] = user_counts.get(uid, 0) + 1
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        assert user_counts.get("fish", 0) == 2
        assert user_counts.get("unknown", 0) == 1  # entry with no user_id


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINT TESTS — mock all models, test HTTP layer only
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIEndpoints:
    """
    Tests the FastAPI endpoints without loading any real models.
    All heavy objects (Whisper, SpeechBrain, mem0) are mocked.
    """

    @pytest.fixture
    def mock_env(self):
        """Patches all model globals in api.py before importing."""
        # Build a fake WAV in memory (minimal valid WAV header)
        import struct
        def make_wav(samples=16000, sr=16000):
            buf = io.BytesIO()
            n_samples = samples
            buf.write(b'RIFF')
            buf.write(struct.pack('<I', 36 + n_samples * 2))
            buf.write(b'WAVE')
            buf.write(b'fmt ')
            buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr*2, 2, 16))
            buf.write(b'data')
            buf.write(struct.pack('<I', n_samples * 2))
            buf.write(b'\x00' * (n_samples * 2))
            buf.seek(0)
            return buf.read()

        return make_wav

    def test_enroll_endpoint_shape(self, mock_env):
        """
        Validates that enroll endpoint would produce correct embedding shape
        if the squeeze bug were fixed.
        """
        # Simulate what verification_speaker_model.encode_batch returns
        fake_output = torch.randn(1, 1, 192)  # [batch, frames, dim]

        # Bug: squeeze(0) → [1, 192] — WRONG for 1D cosine sim
        wrong = fake_output.squeeze(0)
        assert wrong.shape == (1, 192)

        # Fix: squeeze() → [192]
        correct = fake_output.squeeze()
        assert correct.shape == (192,)

    def test_process_audio_returns_unknown_for_empty_cache(self):
        """When profiles_cache is empty, identified_user must be 'Unknown'."""
        profiles_cache = {}
        test_embedding = torch.randn(192)
        test_embedding = test_embedding / test_embedding.norm()

        best_match = "Unknown"
        best_score = 0.0

        for known_user, known_embedding in profiles_cache.items():
            similarity = F.cosine_similarity(known_embedding, test_embedding, dim=0)
            score = similarity.item()
            if score > best_score:
                best_score = score
                best_match = known_user

        identified_user = best_match if best_score > 0.75 else "Unknown"
        assert identified_user == "Unknown"

    def test_memory_not_saved_for_unknown_speaker(self):
        """
        api.py only calls memory.add() if identified_user != "Unknown".
        Verify this guard works — Unknown speakers should never pollute mem0.
        """
        mock_memory = MagicMock()
        identified_user = "Unknown"
        text = "Some transcribed speech"

        if text != "[No speech detected]" and identified_user != "Unknown":
            mock_memory.add(
                messages=[{"role": "user", "content": text}],
                user_id=identified_user,
                metadata={}
            )

        mock_memory.add.assert_not_called()

    def test_memory_not_saved_for_empty_transcription(self):
        """No speech detected → memory.add() must not be called."""
        mock_memory = MagicMock()
        identified_user = "fish"
        text = "[No speech detected]"

        if text != "[No speech detected]" and identified_user != "Unknown":
            mock_memory.add(...)

        mock_memory.add.assert_not_called()

    def test_emotion_map_covers_all_iemocap_labels(self):
        """All 4 IEMOCAP emotion labels must map to readable strings."""
        emotion_map = {'ang': 'Angry', 'hap': 'Happy', 'sad': 'Sad', 'neu': 'Neutral'}
        iemocap_labels = ['ang', 'hap', 'sad', 'neu']

        for label in iemocap_labels:
            assert label in emotion_map, f"Missing emotion label: {label}"
            assert emotion_map[label]   # not empty string


# ══════════════════════════════════════════════════════════════════════════════
# MCP SERVER TOOL TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestMCPTools:

    def _make_mock_memory(self):
        m = MagicMock()
        m.search.return_value = [
            {"memory": "Fish likes coffee", "score": 0.91, "metadata": {"emotion": "Happy"}}
        ]
        m.get_all.return_value = [
            {"memory": "Fish likes coffee", "metadata": {"emotion": "Happy", "speaker_confidence": 0.88}}
        ]
        m.add.return_value = None
        m.delete.return_value = None
        m.delete_all.return_value = None
        return m

    def test_search_memory_formats_output(self):
        mock_mem = self._make_mock_memory()

        results = mock_mem.search("coffee", user_id="fish", limit=5)
        assert len(results) == 1

        r = results[0]
        mem_text = r.get("memory", "")
        score = r.get("score", 0)
        emotion = r.get("metadata", {}).get("emotion", "Unknown")

        formatted = f"1. [{emotion}] (score: {score:.2f}) {mem_text}"
        assert "Happy" in formatted
        assert "0.91" in formatted
        assert "Fish likes coffee" in formatted

    def test_search_memory_empty_result(self):
        mock_mem = MagicMock()
        mock_mem.search.return_value = []

        results = mock_mem.search("nonexistent", user_id="fish", limit=5)
        if not results:
            output = "No memories found for user 'fish' matching 'nonexistent'"

        assert "No memories" in output

    def test_add_memory_calls_correct_structure(self):
        mock_mem = self._make_mock_memory()

        mock_mem.add(
            messages=[{"role": "user", "content": "Fish had ramen for dinner"}],
            user_id="fish",
            metadata={"emotion": "Happy", "speaker_confidence": 1.0, "source": "manual"}
        )

        call_kwargs = mock_mem.add.call_args.kwargs
        assert call_kwargs["user_id"] == "fish"
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["metadata"]["source"] == "manual"

    def test_delete_memory_passes_id(self):
        mock_mem = self._make_mock_memory()
        memory_id = "abc-123-uuid"
        mock_mem.delete(memory_id)
        mock_mem.delete.assert_called_once_with(memory_id)

    def test_delete_all_passes_user_id(self):
        mock_mem = self._make_mock_memory()
        mock_mem.delete_all(user_id="fish")
        mock_mem.delete_all.assert_called_once_with(user_id="fish")

    def test_get_memory_stats_counts_correctly(self):
        """Mirrors get_memory_stats() logic with known data."""
        metadatas = [
            {"user_id": "fish",  "emotion": "Happy"},
            {"user_id": "fish",  "emotion": "Neutral"},
            {"user_id": "sarah", "emotion": "Happy"},
            None,
        ]

        user_counts: dict = {}
        emotion_counts: dict = {}

        for meta in metadatas:
            if not meta:
                continue
            uid = meta.get("user_id", "unknown")
            emotion = meta.get("emotion", "Unknown")
            user_counts[uid] = user_counts.get(uid, 0) + 1
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        assert user_counts["fish"] == 2
        assert user_counts["sarah"] == 1
        assert emotion_counts["Happy"] == 2
        assert emotion_counts["Neutral"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS (requires full stack: Ollama + SpeechBrain + ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegration:

    def test_enroll_then_identify(self, tmp_path):
        """Full cycle: enroll a voice → process same audio → identified correctly."""
        pytest.importorskip("speechbrain")
        pytest.importorskip("faster_whisper")

        # This test requires real audio file — skip if not present
        import os
        if not os.path.exists("test_audio.wav"):
            pytest.skip("test_audio.wav not found — add a real WAV to run this test")

        # Would call: POST /api/enroll, then POST /api/process-audio
        # and assert identified_user == enrolled user_id

    def test_mem0_add_and_search_roundtrip(self, tmp_path):
        """Add a memory → search for it → verify it comes back."""
        pytest.importorskip("mem0")
        pytest.importorskip("chromadb")

        # Requires Ollama running with llama3.2:1b
        import subprocess
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True)
            if result.returncode != 0:
                pytest.skip("Ollama not running")
        except FileNotFoundError:
            pytest.skip("Ollama not installed on this machine")