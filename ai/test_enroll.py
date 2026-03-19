"""
test_enroll.py — Tests for voice enrollment (load_audio_bytes + /api/enroll)

Mocks torchaudio and SpeechBrain so tests run instantly without GPU/ffmpeg.
The conftest.py handles mocking the 'memory' module before import.

Run:
    cd ~/memonic_backend/ai && python -m pytest test_enroll.py -v
"""

import pytest
import torch
import numpy as np
import os
from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════════════════════════
# load_audio_bytes — temp file approach
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadAudioBytes:

    @patch("torchaudio.load")
    def test_writes_temp_file_and_cleans_up(self, mock_torchaudio_load):
        """Temp file is created, torchaudio.load is called with a file path, temp is cleaned up."""
        import models

        # Return a fake signal (mono, 16kHz, 7 seconds)
        fake_signal = torch.randn(1, 16000 * 7)
        mock_torchaudio_load.return_value = (fake_signal, 16000)

        fake_audio_bytes = b"\x00" * 1000

        signal, fs = models.load_audio_bytes(fake_audio_bytes)

        # torchaudio.load was called with a real file path (not BytesIO)
        call_args = mock_torchaudio_load.call_args[0][0]
        assert isinstance(call_args, str), "Expected a file path string, not BytesIO"
        assert call_args.endswith(".wav"), f"Expected .wav suffix, got {call_args}"

        # Temp file should be cleaned up
        assert not os.path.exists(call_args), "Temp file was not deleted after loading"

        # Output is correct
        assert signal.shape == fake_signal.shape
        assert fs == 16000

    @patch("torchaudio.load", side_effect=RuntimeError("Failed to open the input"))
    def test_invalid_audio_raises_error(self, mock_torchaudio_load):
        """Invalid audio bytes → torchaudio.load error propagates, temp still cleaned."""
        import models

        with pytest.raises(RuntimeError, match="Failed to open the input"):
            models.load_audio_bytes(b"not_valid_audio")

        # The temp file path was still cleaned up
        call_args = mock_torchaudio_load.call_args[0][0]
        assert not os.path.exists(call_args), "Temp file was not cleaned up after error"


# ══════════════════════════════════════════════════════════════════════════════
# save_profile — writes .npy and updates cache
# ══════════════════════════════════════════════════════════════════════════════

class TestSaveProfile:

    @patch("numpy.save")
    def test_saves_npy_and_updates_cache(self, mock_np_save, tmp_path):
        """save_profile calls np.save with correct path and updates profiles_cache."""
        import models

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            fake_embedding = torch.randn(192)

            # Mock .cpu().numpy() since torch 2.1 + numpy 2.x are incompatible
            fake_np_array = MagicMock()
            with patch.object(type(fake_embedding), "cpu", return_value=MagicMock(numpy=MagicMock(return_value=fake_np_array))):
                models.save_profile("testuser", fake_embedding)

            # np.save was called with the correct filename
            mock_np_save.assert_called_once()
            save_args = mock_np_save.call_args[0]
            assert save_args[0] == "testuser_profile.npy"
            assert save_args[1] is fake_np_array

            # Cache updated
            assert "testuser" in models.profiles_cache
            assert torch.equal(models.profiles_cache["testuser"], fake_embedding)
        finally:
            models.profiles_cache.pop("testuser", None)
            os.chdir(original_cwd)


# ══════════════════════════════════════════════════════════════════════════════
# /api/enroll — duration validation
# ══════════════════════════════════════════════════════════════════════════════

class TestEnrollEndpoint:
    """
    Tests the /api/enroll endpoint's duration validation logic.
    Mocks load_audio_bytes, encode_speaker, save_profile so no GPU needed.
    """

    def _make_signal(self, duration_seconds, sample_rate=16000):
        n_samples = int(duration_seconds * sample_rate)
        return torch.randn(1, n_samples), sample_rate

    @pytest.fixture
    def enroll_client(self):
        """TestClient for ai/api.py with startup event disabled."""
        from fastapi.testclient import TestClient
        import api

        # Disable the startup event that loads heavy models
        api.app.router.on_startup = []

        with TestClient(api.app) as c:
            yield c

    @patch("models.save_profile")
    @patch("models.encode_speaker", return_value=torch.randn(192))
    @patch("models.load_audio_bytes")
    def test_enroll_too_short(self, mock_load, mock_encode, mock_save, enroll_client):
        """Audio <5s → 400 'Recording length invalid'."""
        mock_load.return_value = self._make_signal(3.0)

        res = enroll_client.post(
            "/api/enroll",
            data={"user_id": "testuser"},
            files={"file": ("test.m4a", b"\x00" * 100, "audio/mp4")},
        )

        assert res.status_code == 400
        assert "invalid" in res.json()["detail"].lower()
        mock_encode.assert_not_called()
        mock_save.assert_not_called()

    @patch("models.save_profile")
    @patch("models.encode_speaker", return_value=torch.randn(192))
    @patch("models.load_audio_bytes")
    def test_enroll_too_long(self, mock_load, mock_encode, mock_save, enroll_client):
        """Audio >12s → 400 'Recording length invalid'."""
        mock_load.return_value = self._make_signal(15.0)

        res = enroll_client.post(
            "/api/enroll",
            data={"user_id": "testuser"},
            files={"file": ("test.m4a", b"\x00" * 100, "audio/mp4")},
        )

        assert res.status_code == 400
        assert "invalid" in res.json()["detail"].lower()
        mock_encode.assert_not_called()
        mock_save.assert_not_called()

    @patch("models.save_profile")
    @patch("models.encode_speaker", return_value=torch.randn(192))
    @patch("models.load_audio_bytes")
    def test_enroll_valid(self, mock_load, mock_encode, mock_save, enroll_client):
        """Audio 7s → 200 + success + correct duration."""
        mock_load.return_value = self._make_signal(7.0)

        res = enroll_client.post(
            "/api/enroll",
            data={"user_id": "testuser"},
            files={"file": ("test.m4a", b"\x00" * 100, "audio/mp4")},
        )

        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "testuser" in data["message"]
        assert data["duration"] == 7.0
        mock_encode.assert_called_once()
        mock_save.assert_called_once()

    @patch("models.save_profile")
    @patch("models.encode_speaker", return_value=torch.randn(192))
    @patch("models.load_audio_bytes")
    def test_enroll_boundary_5s(self, mock_load, mock_encode, mock_save, enroll_client):
        """Audio exactly 5.0s → valid (lower boundary)."""
        mock_load.return_value = self._make_signal(5.0)

        res = enroll_client.post(
            "/api/enroll",
            data={"user_id": "boundary5"},
            files={"file": ("test.m4a", b"\x00" * 100, "audio/mp4")},
        )

        assert res.status_code == 200
        assert res.json()["status"] == "success"

    @patch("models.save_profile")
    @patch("models.encode_speaker", return_value=torch.randn(192))
    @patch("models.load_audio_bytes")
    def test_enroll_boundary_12s(self, mock_load, mock_encode, mock_save, enroll_client):
        """Audio exactly 12.0s → valid (upper boundary)."""
        mock_load.return_value = self._make_signal(12.0)

        res = enroll_client.post(
            "/api/enroll",
            data={"user_id": "boundary12"},
            files={"file": ("test.m4a", b"\x00" * 100, "audio/mp4")},
        )

        assert res.status_code == 200
        assert res.json()["status"] == "success"
