from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import struct
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
import time

try:
    from . import models
    from . import memory
    from .config import (
        DEVICE,
        WHISPER_MODEL_NAME,
        SPEAKER_MODEL_PATH,
        EMOTION_MODEL_PATH,
        CHROMA_PATH,
    )
except Exception:
    # Support running as a script/module without package context
    import models
    import memory
    from config import (
        DEVICE,
        WHISPER_MODEL_NAME,
        SPEAKER_MODEL_PATH,
        EMOTION_MODEL_PATH,
        CHROMA_PATH,
    )

app = FastAPI()

# ── In-memory device status (updated by ESP32, read by the app) ──
device_state = {"bracelet_last_seen": 0.0}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Initialize heavy models and memory clients asynchronously
    await models.init_models(device=DEVICE,
                              whisper_name=WHISPER_MODEL_NAME,
                              speaker_path=SPEAKER_MODEL_PATH,
                              emotion_path=EMOTION_MODEL_PATH)
    await memory.init_memory(chroma_path=CHROMA_PATH)
    models.load_all_profiles()
    # start silence watcher
    import asyncio
class EnrollRequest(BaseModel):
    user_id: str
    file_ext: str
    audio_base64: str

class ProcessAudioRequest(BaseModel):
    file_ext: str
    audio_base64: str


@app.post("/api/enroll")
async def enroll_voice(req: EnrollRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        
        # ส่ง filename เป็น dummy name + file_ext ขึ้นมา
        filename = f"audio{req.file_ext}"
        signal, fs = models.load_audio_bytes(audio_bytes, filename)
        
        duration_seconds = signal.shape[1] / fs
        if duration_seconds < 5.0 or duration_seconds > 12.0:
            raise HTTPException(status_code=400, detail="Recording length invalid")

        embedding = models.encode_speaker(signal)
        models.save_profile(req.user_id, embedding)

        return {"status": "success", "message": f"User '{req.user_id}' enrolled.", "duration": round(duration_seconds,2)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/members_voice")
async def get_members_voice():
    """Return list of enrolled member names from member_voice/ folder."""
    import os, glob
    member_voice_dir = "member_voice"
    if not os.path.exists(member_voice_dir):
        return []
    npy_files = glob.glob(os.path.join(member_voice_dir, "*_profile.npy"))
    members = []
    for f in sorted(npy_files):
        name = os.path.basename(f).replace("_profile.npy", "")
        members.append(name)
    return members

@app.post("/api/process-audio")
async def process_audio(req: ProcessAudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        filename = f"audio{req.file_ext}"
        signal, fs = models.load_audio_bytes(audio_bytes, filename)
        if signal.shape[0] > 1:
            signal = models.to_mono(signal)

        # speaker identification
        test_embedding = models.encode_speaker(signal)
        identified_user, best_score = models.match_speaker(test_embedding)

        # emotion
        detected_emotion = models.classify_emotion(signal)

        # transcription
        text = models.transcribe(signal)
        if not text:
            text = "[No speech detected]"

        if text != "[No speech detected]" and identified_user != "Unknown":
            memory.save_memory(identified_user, text, detected_emotion, best_score)

        return {
            "status": "success",
            "identified_user": identified_user,
            "speaker_confidence": round(best_score, 4),
            "emotion": detected_emotion,
            "transcription": text,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# SHARED AUDIO PIPELINE — used by both HTTP and WebSocket
# ══════════════════════════════════════════════════════════════
def build_wav_header(data_size: int, sample_rate: int = 16000,
                     bits_per_sample: int = 16, channels: int = 1) -> bytes:
    """Build a 44-byte WAV header for raw PCM data."""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')
    header += struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, channels,
                          sample_rate, byte_rate, block_align, bits_per_sample)
    header += struct.pack('<4sI', b'data', data_size)
    return header


async def process_audio_pipeline(wav_bytes: bytes, source: str = "HTTP") -> dict:
    """
    Shared AI pipeline for both HTTP POST and WebSocket audio.
    Returns dict with: identified_user, speaker_confidence, emotion, transcription, summary
    """
    import os, torch
    from datetime import datetime

    device_state["bracelet_last_seen"] = time.time()

    t_start = time.time()
    print("\n" + "=" * 60)
    print(f"🎤 ESP32 AUDIO RECEIVED ({source})")
    print("=" * 60)

    duration_est = (len(wav_bytes) - 44) / (16000 * 2)
    print(f"📦 Size: {len(wav_bytes):,} bytes (~{duration_est:.1f}s audio)")

    # Save debug audio
    debug_dir = "debug_audio"
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = os.path.join(debug_dir, f"esp32_{timestamp}.wav")
    with open(debug_path, "wb") as f:
        f.write(wav_bytes)
    print(f"💾 Debug audio saved: {debug_path}")

    signal, fs = models.load_audio_bytes(wav_bytes, "audio.wav")
    if signal.shape[0] > 1:
        signal = models.to_mono(signal)
    print(f"✅ Audio loaded: {signal.shape}, sample rate: {fs}")

    # Signal stats
    audio_flat = signal.squeeze()
    rms = torch.sqrt(torch.mean(audio_flat ** 2)).item()
    peak = torch.max(torch.abs(audio_flat)).item()
    print(f"📊 Signal stats — RMS: {rms:.6f}, Peak: {peak:.6f}, Samples: {audio_flat.shape[0]}")
    if rms < 0.001:
        print("⚠️  WARNING: Audio is nearly SILENT — mic may not be working!")

    # Speaker identification
    t1 = time.time()
    print("🔍 Running speaker identification...")
    test_embedding = models.encode_speaker(signal)
    identified_user, best_score = models.match_speaker(test_embedding)
    print(f"   → Speaker: {identified_user} (confidence: {best_score:.4f}) [{time.time()-t1:.2f}s]")

    # Emotion detection
    t2 = time.time()
    print("😊 Running emotion detection...")
    detected_emotion = models.classify_emotion(signal)
    print(f"   → Emotion: {detected_emotion} [{time.time()-t2:.2f}s]")

    # Transcription
    t3 = time.time()
    print("📝 Running transcription...")
    text = models.transcribe(signal)
    if not text:
        text = "[No speech detected]"
    print(f"   → Text: {text} [{time.time()-t3:.2f}s]")

    # Terminal popup
    print()
    print("┌" + "─" * 58 + "┐")
    print(f"│  [{identified_user}[{best_score:.2f}]][{detected_emotion}]: {text[:48]}")
    if len(text) > 48:
        print(f"│  {text[48:]}")
    print("└" + "─" * 58 + "┘")

    summary = ""
    if text != "[No speech detected]" and identified_user != "Unknown":
        t4 = time.time()
        print("💾 Saving to ChromaDB...")
        memory.save_memory(identified_user, text, detected_emotion, best_score)
        print(f"   → Saved [{time.time()-t4:.2f}s]")

        t5 = time.time()
        print("🤖 Generating LLM summary...")
        summary = await memory.summarize_and_popup(identified_user, text, detected_emotion)
        print(f"   → Summary: {summary} [{time.time()-t5:.2f}s]")
        print(f"📱 Popup stored for '{identified_user}' → mobile app will pick it up")
    else:
        if identified_user == "Unknown":
            print("⚠️  Unknown speaker — memory NOT saved")
        if text == "[No speech detected]":
            print("⚠️  No speech detected — memory NOT saved")

    total_time = time.time() - t_start
    print(f"\n⏱️  Total processing time: {total_time:.2f}s")
    print("=" * 60 + "\n")

    return {
        "identified_user": identified_user,
        "speaker_confidence": round(best_score, 4),
        "emotion": detected_emotion,
        "transcription": text,
        "summary": summary,
    }


@app.post("/api/esp32-audio")
async def esp32_audio(request: Request):
    """
    Receive raw WAV bytes from the ESP32 bracelet via HTTP POST (legacy).
    For better reliability, use the WebSocket endpoint /ws/audio instead.
    """
    try:
        audio_bytes = await request.body()
        if len(audio_bytes) < 44:
            raise HTTPException(status_code=400, detail="Audio too short (need WAV with header)")
        result = await process_audio_pipeline(audio_bytes, source="HTTP POST")
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("❌ ERROR in esp32_audio:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Debug Audio Playback ─────────────────────────────────
@app.get("/api/debug-audio")
async def list_debug_audio():
    """List all saved debug audio files. Open any URL in your browser to hear it."""
    import os, glob
    debug_dir = "debug_audio"
    if not os.path.exists(debug_dir):
        return {"files": [], "message": "No debug audio yet. Send audio from ESP32 first."}

    files = sorted(glob.glob(os.path.join(debug_dir, "*.wav")), reverse=True)
    base_url = "https://8001-01kkh2et3bdjymj2fjq6jabg8k.cloudspaces.litng.ai"
    result = []
    for f in files:
        filename = os.path.basename(f)
        size_kb = os.path.getsize(f) / 1024
        result.append({
            "filename": filename,
            "size_kb": round(size_kb, 1),
            "play_url": f"{base_url}/api/debug-audio/{filename}",
        })
    return {"files": result, "message": f"Open any play_url in your browser to hear the audio."}


@app.get("/api/debug-audio/{filename}")
async def get_debug_audio(filename: str):
    """Serve a debug audio WAV file — open in browser to play it."""
    import os
    file_path = os.path.join("debug_audio", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)


# ══════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINT — ESP32 connects once, streams continuously
#
# Protocol:
#   ESP32 → Server:  TEXT "START"     (begin recording)
#   ESP32 → Server:  BINARY chunks    (raw 16-bit PCM at 16kHz)
#   ESP32 → Server:  TEXT "STOP"      (end recording, triggers processing)
#   Server → ESP32:  TEXT JSON        (result: speaker, emotion, text, summary)
#
# Benefits over HTTP POST:
#   - Single persistent connection (no reconnect overhead)
#   - Auto-reconnect if WiFi drops
#   - Much more reliable for streaming audio
# ══════════════════════════════════════════════════════════════
@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()
    is_recording = False
    chunks_received = 0

    print("\n🔌 WebSocket client connected")
    print(f"   Free to stream audio. Waiting for START command...")

    try:
        while True:
            msg = await websocket.receive()

            if msg["type"] == "websocket.disconnect":
                break

            # ── Text commands: START / STOP ──
            if "text" in msg:
                command = msg["text"].strip().upper()

                if command == "START":
                    audio_buffer = bytearray()
                    chunks_received = 0
                    is_recording = True
                    print("🎤 WebSocket: Recording STARTED")

                elif command == "STOP":
                    is_recording = False
                    data_size = len(audio_buffer)
                    duration = data_size / (16000 * 2)
                    print(f"⏹️  WebSocket: Recording STOPPED — {data_size:,} bytes ({duration:.1f}s), {chunks_received} chunks")

                    if data_size < 1600:  # less than 0.05s of audio
                        await websocket.send_json({"error": "Audio too short"})
                        continue

                    # Build WAV from raw PCM buffer
                    wav_header = build_wav_header(data_size, sample_rate=16000)
                    wav_bytes = wav_header + bytes(audio_buffer)

                    try:
                        result = await process_audio_pipeline(wav_bytes, source="WebSocket")
                        await websocket.send_json(result)
                    except Exception as e:
                        import traceback
                        print(f"❌ Pipeline error: {e}")
                        traceback.print_exc()
                        await websocket.send_json({"error": str(e)})

            # ── Binary data: raw PCM audio chunks ──
            elif "bytes" in msg and is_recording:
                audio_buffer.extend(msg["bytes"])
                chunks_received += 1

    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")


@app.get("/api/check-popup/{user_id}")
async def check_popup(user_id: str):
    return memory.check_popup(user_id)


@app.post("/api/trigger-summary/{user_id}")
async def trigger_summary(user_id: str):
    return await memory.trigger_summary(user_id)


@app.get("/api/get-home-data/{user_id}")
async def get_home_data(user_id: str):
    return await memory.get_home_data(user_id)


@app.get("/api/get-events/{user_id}")
async def get_events(user_id: str):
    return await memory.get_events(user_id)


@app.get("/api/get-mood/{user_id}")
async def get_mood(user_id: str):
    return await memory.get_mood(user_id)


@app.get("/api/members")
async def list_members():
    """List all users who have enrolled their voice."""
    import os, glob
    member_voice_dir = "member_voice"
    if not os.path.exists(member_voice_dir):
        return []
    
    npy_files = glob.glob(os.path.join(member_voice_dir, "*_profile.npy"))
    members = []
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        user_id = filename.replace("_profile.npy", "")
        members.append(user_id)
    
    return sorted(members)


@app.delete("/api/voice-profile/{user_id}")
async def delete_voice_profile(user_id: str):
    import os
    file_path = f"member_voice/{user_id}_profile.npy"
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "success", "message": f"Voice profile for '{user_id}' deleted."}
    else:
        raise HTTPException(status_code=404, detail="Voice profile not found")


# ── ESP32 Device Status ──────────────────────────────────────────
class DeviceStatusUpdate(BaseModel):
    bracelet: str = "Connected"
    dock: str = "Connected"

@app.post("/update")
async def update_device(data: DeviceStatusUpdate):
    """ESP32 calls this to push bracelet & dock status."""
    device_state["bracelet_last_seen"] = time.time()
    return {"status": "ok"}

@app.get("/device-status")
async def get_device_status():
    """App calls this to read latest ESP32 values."""
    # If the ESP32 pinged in the last 30 seconds, it's alive.
    is_alive = (time.time() - device_state["bracelet_last_seen"]) < 30
    

    return {
        "bracelet": "Connected" if is_alive else "Disconnected",
        "dock": "Connected"
    }


# ── Test / Dev Helper ────────────────────────────────────────────
class SeedMemoryRequest(BaseModel):
    user_id: str = "fish"
    entries: list = [
        "I have a meeting with my professor tomorrow at 3 PM about the final project.",
        "I need to pick up groceries after class. We're out of milk and eggs.",
        "I'm feeling pretty stressed about the upcoming exam on Friday.",
        "My friend invited me to play basketball at the gym this Saturday at 10 AM.",
        "I should call mom tonight. It's been a while since we talked.",
    ]
    emotions: list = ["Neutral", "Happy", "Angry", "Happy", "Sad"]

@app.post("/api/test-seed-memory")
async def test_seed_memory(req: SeedMemoryRequest):
    """
    DEV ONLY - Seed fake memories into ChromaDB so the dashboard populates.
    Delete this endpoint before production!
    """
    saved = 0
    for i, text in enumerate(req.entries):
        emotion = req.emotions[i] if i < len(req.emotions) else "Neutral"
        try:
            memory.save_memory(req.user_id, text, emotion, 0.95)
            saved += 1
        except Exception as e:
            print(f"seed error: {e}")
    return {"status": "ok", "saved": saved, "user_id": req.user_id}
# ─────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)