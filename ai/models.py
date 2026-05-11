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


def preprocess_audio(signal: torch.Tensor, fs: int = 16000) -> torch.Tensor:
    """
    Clean up raw audio before encoding for speaker recognition.
    Steps:
      1. Mono mix
      2. Resample to 16kHz (ECAPA expects this)
      3. High-pass filter @ 80Hz — removes DC bias, hum, low rumble
      4. Trim leading/trailing silence (energy-based VAD)
      5. RMS normalize to consistent loudness
    """
    # 1. Mono
    if signal.dim() == 2 and signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)

    # 2. Resample if needed
    if fs != 16000:
        signal = torchaudio.transforms.Resample(fs, 16000)(signal)
        fs = 16000

    # 3. High-pass filter @ 80Hz
    try:
        signal = torchaudio.functional.highpass_biquad(signal, fs, cutoff_freq=80.0)
    except Exception:
        pass  # not fatal if filter fails

    # 4. Trim silence — frame-based RMS, drop frames below -40dB of peak
    try:
        frame = int(0.02 * fs)  # 20ms frames
        flat = signal.squeeze(0)
        if flat.numel() > frame * 3:
            n_frames = flat.numel() // frame
            energies = flat[: n_frames * frame].reshape(n_frames, frame).pow(2).mean(dim=1).sqrt()
            peak = float(energies.max().item())
            if peak > 1e-6:
                threshold = peak * 0.10  # -20dB below peak
                voiced = (energies > threshold).nonzero(as_tuple=True)[0]
                if voiced.numel() > 0:
                    start = int(voiced[0].item()) * frame
                    end = (int(voiced[-1].item()) + 1) * frame
                    signal = signal[:, start:end]
    except Exception:
        pass

    # 5. RMS normalize to target -20dBFS (rms ≈ 0.1)
    rms = signal.pow(2).mean().sqrt()
    if float(rms.item()) > 1e-6:
        signal = signal * (0.1 / rms)
        # Clip to prevent overflow (shouldn't happen after RMS norm but safe)
        signal = signal.clamp(-1.0, 1.0)

    return signal


def encode_speaker(signal: torch.Tensor, preprocess: bool = True, fs: int = 16000) -> torch.Tensor:
    """Encode audio → speaker embedding. Applies preprocessing by default."""
    if _speaker is None:
        raise RuntimeError("Speaker model not loaded")
    if preprocess:
        signal = preprocess_audio(signal, fs)
    emb = _speaker.encode_batch(signal).squeeze()
    return emb


def save_profile(user_id: str, embedding: torch.Tensor):
    """
    Save a speaker profile.
    If user_id already exists, the new embedding is averaged with the old one
    (multi-sample centroid). This makes enrollment more robust — call this
    function multiple times with different samples of the same voice.
    """
    member_voice_dir = "member_voice"
    if not os.path.exists(member_voice_dir):
        os.makedirs(member_voice_dir, exist_ok=True)

    new_emb = embedding.detach().cpu()
    # L2-normalize before averaging so each sample contributes equally
    new_emb = new_emb / (new_emb.norm() + 1e-9)

    # Counter tracks how many samples have been averaged
    count_path = os.path.join(member_voice_dir, f"{user_id}_count.txt")
    file_path  = os.path.join(member_voice_dir, f"{user_id}_profile.npy")

    if os.path.exists(file_path) and os.path.exists(count_path):
        try:
            n = int(open(count_path).read().strip())
            old = torch.from_numpy(np.load(file_path))
            # Running mean: new_centroid = (old * n + new) / (n + 1)
            centroid = (old * n + new_emb) / (n + 1)
            centroid = centroid / (centroid.norm() + 1e-9)
            n += 1
        except Exception:
            centroid = new_emb
            n = 1
    else:
        centroid = new_emb
        n = 1

    np.save(file_path, centroid.numpy())
    with open(count_path, "w") as f:
        f.write(str(n))
    profiles_cache[user_id] = centroid


def reset_profile(user_id: str):
    """Delete an enrollment so the next save_profile starts fresh."""
    member_voice_dir = "member_voice"
    for suffix in ("_profile.npy", "_count.txt"):
        p = os.path.join(member_voice_dir, f"{user_id}{suffix}")
        if os.path.exists(p):
            os.remove(p)
    profiles_cache.pop(user_id, None)


def profile_sample_count(user_id: str) -> int:
    """How many samples have been averaged into this user's profile."""
    p = os.path.join("member_voice", f"{user_id}_count.txt")
    try:
        return int(open(p).read().strip())
    except Exception:
        return 0


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

    # ECAPA-TDNN cosine similarity:
    #   same speaker  → 0.45-0.85 (depends heavily on mic + noise + clip length)
    #   different     → 0.0-0.35
    # 0.45 is the canonical practical threshold for this model.
    identified_user = best_match if best_score > 0.45 else "Unknown"
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
