import io
import glob
import os
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F

# ── SpeechBrain / PyTorch compatibility patch ────────────────────
# SpeechBrain 0.5.x calls torch.amp.custom_fwd(device_type=...)
# • PyTorch < 2.4  → torch.amp.custom_fwd doesn't exist at all
# • PyTorch ≥ 2.4  → torch.amp.custom_fwd exists but REJECTS device_type
# This wrapper bridges both directions.
import inspect as _inspect

if not hasattr(torch.amp, 'custom_fwd'):
    # OLD PyTorch: custom_fwd lives under torch.cuda.amp
    _orig_custom_fwd = torch.cuda.amp.custom_fwd
    _orig_custom_bwd = torch.cuda.amp.custom_bwd

    def _compat_custom_fwd(fn=None, *, device_type=None, cast_inputs=None):
        return _orig_custom_fwd(fn, cast_inputs=cast_inputs)

    def _compat_custom_bwd(fn=None, *, device_type=None):
        return _orig_custom_bwd(fn)

    torch.amp.custom_fwd = _compat_custom_fwd
    torch.amp.custom_bwd = _compat_custom_bwd

elif 'device_type' not in _inspect.signature(torch.amp.custom_fwd).parameters:
    # NEW PyTorch 2.4+: custom_fwd exists but no longer accepts device_type
    _orig_custom_fwd = torch.amp.custom_fwd
    _orig_custom_bwd = torch.amp.custom_bwd

    def _compat_custom_fwd(fn=None, *, device_type=None, cast_inputs=None):
        if cast_inputs is not None:
            return _orig_custom_fwd(fn, cast_inputs=cast_inputs)
        return _orig_custom_fwd(fn) if fn else _orig_custom_fwd

    def _compat_custom_bwd(fn=None, *, device_type=None):
        return _orig_custom_bwd(fn) if fn else _orig_custom_bwd

    torch.amp.custom_fwd = _compat_custom_fwd
    torch.amp.custom_bwd = _compat_custom_bwd
# ────────────────────────────────────────────────────────────────

from faster_whisper import WhisperModel
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.inference.classifiers import EncoderClassifier
from typing import Tuple

# Lightweight model loader and helpers
_device = "cpu"
_whisper = None
_speaker = None
_emotion = None
profiles_cache: dict = {}


async def init_models(device: str = "cpu", whisper_name: str = "small",
                      speaker_path: str = "pretrained_models/spkrec-ecapa-voxceleb",
                      emotion_path: str = "pretrained_models/emotion-recognition-wav2vec2-IEMOCAP"):
    global _device
    global _whisper, _speaker, _emotion
    _device = device
    try:
        _whisper = WhisperModel(whisper_name, device=device, compute_type="float32")
    except Exception:
        _whisper = None

    try:
        _speaker = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=speaker_path,
            run_opts={"device": device}
        )
    except Exception:
        _speaker = None

    try:
        _emotion = EncoderClassifier.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir=emotion_path,
            run_opts={"device": device}
        )
    except Exception:
        _emotion = None


def load_all_profiles():
    # Load any saved *_profile.npy files from member_voice folder into RAM cache
    member_voice_dir = "member_voice"
    if not os.path.exists(member_voice_dir):
        return
    npy_files = glob.glob(os.path.join(member_voice_dir, "*_profile.npy"))
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        user_id = filename.replace("_profile.npy", "")
        enrolled_np = np.load(file_path)
        profiles_cache[user_id] = torch.from_numpy(enrolled_np).to(_device)


def load_audio_bytes(audio_bytes: bytes, filename: str = "audio.wav") -> Tuple[torch.Tensor, int]:
    import tempfile, os
    
    # ดึงนามสกุลไฟล์ออกมาใช้งาน (เพื่อรองรับทั้ง .m4a, .mp3, .wav ฯลฯ)
    ext = os.path.splitext(filename)[1]
    if not ext:
        ext = ".wav"
        
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        signal, fs = torchaudio.load(tmp_path)
    finally:
        os.unlink(tmp_path)
    return signal, fs


def to_mono(signal: torch.Tensor) -> torch.Tensor:
    return torch.mean(signal, dim=0, keepdim=True)


def encode_speaker(signal: torch.Tensor) -> torch.Tensor:
    if _speaker is None:
        raise RuntimeError("Speaker model not loaded")
    emb = _speaker.encode_batch(signal).squeeze()
    return emb


def save_profile(user_id: str, embedding: torch.Tensor):
    # Save .npy files to member_voice folder
    member_voice_dir = "member_voice"
    if not os.path.exists(member_voice_dir):
        os.makedirs(member_voice_dir, exist_ok=True)
    
    emb_np = embedding.detach().cpu().numpy()
    file_path = os.path.join(member_voice_dir, f"{user_id}_profile.npy")
    np.save(file_path, emb_np)
    profiles_cache[user_id] = embedding


def match_speaker(test_embedding: torch.Tensor):
    # keep same logic as original: floor at 0.0 so negative scores won't override
    best_match = "Unknown"
    best_score = 0.0
    for known_user, known_embedding in profiles_cache.items():
        # ensure 1D embeddings
        if known_embedding.dim() > 1:
            known = known_embedding.squeeze()
        else:
            known = known_embedding
        if test_embedding.dim() > 1:
            test = test_embedding.squeeze()
        else:
            test = test_embedding

        similarity = F.cosine_similarity(known.unsqueeze(0), test.unsqueeze(0), dim=1)
        score = float(similarity.item())
        if score > best_score:
            best_score = score
            best_match = known_user

    identified_user = best_match if best_score > 0.75 else "Unknown"
    return identified_user, best_score


def classify_emotion(signal: torch.Tensor) -> str:
    if _emotion is None:
        return "Unknown"
    try:
        emotion_signal = signal.squeeze(0).unsqueeze(0)
        out_prob, score, index, text_lab = _emotion.classify_batch(emotion_signal)
        emotion_map = {'ang': 'Angry', 'hap': 'Happy', 'sad': 'Sad', 'neu': 'Neutral'}
        return emotion_map.get(text_lab[0], 'Unknown')
    except Exception:
        return "Unknown"


def transcribe(signal: torch.Tensor) -> str:
    if _whisper is None:
        return ""
    signal_np = signal.squeeze().cpu().numpy()
    segments, info = _whisper.transcribe(signal_np, beam_size=5, language="en")
    text = " ".join([seg.text for seg in list(segments)]).strip()
    return text
