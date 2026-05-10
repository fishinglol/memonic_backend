import os
import glob
import tempfile
import numpy as np
import time
import torch
import torchaudio
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, FastAPI, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
import logging
from pydantic import BaseModel
from typing import List

from core.database import SessionLocal
from core.models import Memory
import ai.memory as memory
import ai.models as models

# --- State Management ---
device_state = {"bracelet_last_seen": 0}

# Bracelet WS connection + current job state
# job state machine: idle → recording → processing → success/error → idle
bracelet_state = {
    "ws": None,
    "last_seen": 0,
    "job": {
        "state": "idle",       # idle | recording | processing | success | error
        "mode": None,          # ENROLL | START
        "user": None,          # name being enrolled
        "result": None,        # final text response from pipeline
        "started_at": 0,
    },
}

def _reset_job():
    bracelet_state["job"] = {
        "state": "idle",
        "mode": None,
        "user": None,
        "result": None,
        "started_at": 0,
    }

logger = logging.getLogger(__name__)

router = APIRouter()
app = FastAPI()
# NOTE: app.include_router(router) is called at the END of this file,
# AFTER all @router.* decorators have registered their routes.

# Initialize AI models (Whisper, Speaker, Emotion)
# This will use the logic in ai/models.py
try:
    import asyncio
    # Run initialization in the background or at startup
    # For now, we'll call it synchronously to ensure they are ready
    # Note: In a real production app, you might want to do this asynchronously
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(models.init_models())
    else:
        asyncio.run(models.init_models())
    models.load_all_profiles()
except Exception as e:
    logger.error(f"Failed to initialize AI models: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def identify_speaker(audio_path: str) -> str:
    try:
        signal, fs = torchaudio.load(audio_path)
        if signal.shape[0] > 1:
            signal = models.to_mono(signal)
        
        # Resample to 16kHz if needed (models usually expect 16kHz)
        if fs != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(fs, 16000)
            signal = resampler(signal)
            
        embedding = models.encode_speaker(signal)
        best_match, score = models.match_speaker(embedding)
        return best_match
    except Exception as e:
        logger.error(f"Speaker identification failed: {e}")
        return "unknown"

def get_emotion(audio_path: str) -> str:
    try:
        signal, fs = torchaudio.load(audio_path)
        return models.classify_emotion(signal)
    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
    return "unknown"

def transcribe_audio(audio_path: str) -> str:
    try:
        signal, fs = torchaudio.load(audio_path)
        return models.transcribe(signal)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return ""


@router.post("/audio")
async def process_audio(file: UploadFile = File(...), enroll_user: str = Form(None), db: Session = Depends(get_db)):
    # Update heartbeat
    device_state["bracelet_last_seen"] = time.time()
    # Validate file size
    if not file:
        raise HTTPException(status_code=400, detail={"status": "error", "message": "No file provided"})
        
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size <= 1000:
        raise HTTPException(status_code=400, detail={"status": "error", "message": "File too small, needs to be > 1000 bytes"})

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        if enroll_user:
            try:
                signal, fs = torchaudio.load(temp_file.name)
                embedding = models.encode_speaker(signal)
                models.save_profile(enroll_user, embedding)
                return {
                    "status": "ok",
                    "text": f"SUCCESS: Enrolled {enroll_user}",
                    "memory_saved": False
                }
            except Exception as e:
                logger.error(f"Enroll error: {e}")
                raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})
        
        # Step 2: Transcribe
        transcript = transcribe_audio(temp_file.name)

        # Step 3: Speaker ID
        speaker = identify_speaker(temp_file.name)
        
        # Step 4: Emotion
        emotion = get_emotion(temp_file.name)
        
        # Step 5: Save to SQLite table "memories"
        timestamp = datetime.utcnow()
        new_memory = Memory(
            transcript=transcript,
            speaker=speaker,
            emotion=emotion,
            timestamp=timestamp
        )
        db.add(new_memory)
        db.commit()
        db.refresh(new_memory)
        
        # Step 6: Trigger Popup (summarization + push notification)
        try:
            user_id_for_popup = speaker if speaker != "unknown" else "user"
            asyncio.create_task(memory.summarize_and_popup(user_id_for_popup, transcript, emotion))
        except Exception as e:
            logger.error(f"Popup trigger failed: {e}")

        # Step 7: Return JSON
        return {
            "status": "ok",
            "transcript": transcript,
            "speaker": speaker,
            "emotion": emotion,
            "memory_saved": True,
            "timestamp": timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


@router.post("/api/enroll")
async def enroll_member(req: dict):
    """Enroll a new member via base64 audio (used by AddMemberSheet)."""
    user_id = req.get("user_id")
    audio_base64 = req.get("audio_base_64")
    file_ext = req.get("file_ext", ".m4a")

    if not user_id or not audio_base64:
        raise HTTPException(status_code=400, detail="Missing user_id or audio_base_64")

    import base64
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    try:
        audio_bytes = base64.b64decode(audio_base64)
        with open(temp_file.name, "wb") as f:
            f.write(audio_bytes)
        
        signal, fs = torchaudio.load(temp_file.name)
        embedding = models.encode_speaker(signal)
        models.save_profile(user_id, embedding)
        
        return {"status": "ok", "message": f"SUCCESS: Enrolled {user_id}"}
    except Exception as e:
        logger.error(f"Enroll error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


@router.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket, db: Session = Depends(get_db)):
    """
    Server-driven bracelet channel.
    Bracelet connects and waits. Server sends 'ENROLL <name>' or 'START' (triggered
    by /api/bracelet/enroll or /api/bracelet/record). Bracelet then streams binary
    PCM frames followed by a 'STOP' text frame.
    """
    await websocket.accept()
    bracelet_state["ws"] = websocket
    bracelet_state["last_seen"] = time.time()
    device_state["bracelet_last_seen"] = time.time()
    logger.info("Bracelet WS connected")

    audio_data = bytearray()

    try:
        while True:
            msg = await websocket.receive()
            now = time.time()
            bracelet_state["last_seen"] = now
            device_state["bracelet_last_seen"] = now

            # Client disconnected cleanly — break instead of calling receive() again
            if msg.get("type") == "websocket.disconnect":
                logger.info("Bracelet sent disconnect frame")
                break

            if "bytes" in msg:
                if bracelet_state["job"]["state"] == "recording":
                    audio_data.extend(msg["bytes"])
                continue

            if "text" not in msg:
                continue

            txt = msg["text"].strip()
            up = txt.upper()

            # Bracelet greets us — ignore
            if up == "HELLO":
                continue

            # End of stream — process
            if up == "STOP" and bracelet_state["job"]["state"] == "recording":
                job = bracelet_state["job"]
                is_enroll = (job["mode"] == "ENROLL")
                enroll_user = job["user"]

                if len(audio_data) < 1000:
                    job["state"] = "error"
                    job["result"] = "ERROR: Audio too short"
                    await websocket.send_text(job["result"])
                    audio_data = bytearray()
                    continue

                job["state"] = "processing"
                import wave
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                try:
                    with wave.open(temp_file.name, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(16000)
                        wav_file.writeframes(audio_data)

                    if is_enroll and enroll_user:
                        try:
                            signal, fs = torchaudio.load(temp_file.name)
                            embedding = models.encode_speaker(signal)
                            models.save_profile(enroll_user, embedding)
                            result = f"SUCCESS: Enrolled {enroll_user}"
                            job["state"] = "success"
                        except Exception as e:
                            logger.error(f"Enroll error: {e}")
                            result = f"ERROR: {e}"
                            job["state"] = "error"
                    else:
                        transcript = transcribe_audio(temp_file.name)
                        speaker = identify_speaker(temp_file.name)
                        emotion = get_emotion(temp_file.name)
                        new_memory = Memory(
                            transcript=transcript,
                            speaker=speaker,
                            emotion=emotion,
                            timestamp=datetime.utcnow(),
                        )
                        db.add(new_memory)
                        db.commit()
                        db.refresh(new_memory)
                        try:
                            user_id_for_popup = speaker if speaker != "unknown" else "user"
                            asyncio.create_task(memory.summarize_and_popup(user_id_for_popup, transcript, emotion))
                        except Exception as e:
                            logger.error(f"Popup error: {e}")
                        result = f"OK: {transcript} | Speaker: {speaker} | Emotion: {emotion}"
                        job["state"] = "success"

                    job["result"] = result
                    await websocket.send_text(result)
                finally:
                    if os.path.exists(temp_file.name):
                        os.remove(temp_file.name)
                audio_data = bytearray()
                continue

    except WebSocketDisconnect:
        logger.info("Bracelet WS disconnected")
    except Exception as e:
        logger.error(f"Bracelet WS error: {e}")
        try:
            await websocket.send_text(f"ERROR: {str(e)}")
        except Exception:
            pass
    finally:
        if bracelet_state["ws"] is websocket:
            bracelet_state["ws"] = None
        if bracelet_state["job"]["state"] in ("recording", "processing"):
            bracelet_state["job"]["state"] = "error"
            bracelet_state["job"]["result"] = "ERROR: Bracelet disconnected"


# ── HTTP control endpoints (called by UI) ────────────────────────

class EnrollRequest(BaseModel):
    name: str

@router.post("/api/bracelet/enroll")
async def bracelet_enroll(req: EnrollRequest):
    """UI triggers enrollment. Server pushes 'ENROLL <name>' to bracelet over WSS."""
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    ws = bracelet_state["ws"]
    if ws is None:
        raise HTTPException(status_code=503, detail="Bracelet not connected")
    if bracelet_state["job"]["state"] in ("recording", "processing"):
        raise HTTPException(status_code=409, detail="Bracelet busy")

    bracelet_state["job"] = {
        "state": "recording",
        "mode": "ENROLL",
        "user": name,
        "result": None,
        "started_at": time.time(),
    }
    try:
        await ws.send_text(f"ENROLL {name}")
    except Exception as e:
        bracelet_state["job"]["state"] = "error"
        bracelet_state["job"]["result"] = f"ERROR: {e}"
        raise HTTPException(status_code=502, detail=str(e))
    return {"ok": True, "name": name}


@router.post("/api/bracelet/record")
async def bracelet_record(seconds: int = 5):
    """
    UI manually triggers a normal memory recording.
    `seconds` is the recording duration (clamped 1..30).
    Sent to bracelet as 'START <n>'.
    """
    seconds = max(1, min(int(seconds), 30))
    ws = bracelet_state["ws"]
    if ws is None:
        raise HTTPException(status_code=503, detail="Bracelet not connected")
    if bracelet_state["job"]["state"] in ("recording", "processing"):
        raise HTTPException(status_code=409, detail="Bracelet busy")
    bracelet_state["job"] = {
        "state": "recording",
        "mode": "START",
        "user": None,
        "result": None,
        "started_at": time.time(),
    }
    try:
        await ws.send_text(f"START {seconds}")
    except Exception as e:
        bracelet_state["job"]["state"] = "error"
        bracelet_state["job"]["result"] = f"ERROR: {e}"
        raise HTTPException(status_code=502, detail=str(e))
    return {"ok": True, "seconds": seconds}


@router.get("/api/bracelet/status")
async def bracelet_status():
    """UI polls this to know connection + current job result."""
    online = bracelet_state["ws"] is not None and (time.time() - bracelet_state["last_seen"]) < 30
    return {
        "online": online,
        "last_seen": bracelet_state["last_seen"],
        "job": bracelet_state["job"],
    }


@router.post("/api/bracelet/reset")
async def bracelet_reset():
    """UI clears the job state (e.g. after reading the result)."""
    _reset_job()
    return {"ok": True}


@router.get("/api/memories")
def get_memories(limit: int = 20, db: Session = Depends(get_db)):
    limit = min(max(1, limit), 100)
    try:
        memories = db.query(Memory).order_by(Memory.timestamp.desc()).limit(limit).all()
        return {
            "memories": [
                {
                    "id": m.id,
                    "transcript": m.transcript,
                    "speaker": m.speaker,
                    "emotion": m.emotion,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None
                }
                for m in memories
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching memories: {e}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})


# ── DASHBOARD & AI ENDPOINTS ──────────────────────────────────────

@router.get("/api/check-popup/{user_id}")
async def check_popup(user_id: str):
    return memory.check_popup(user_id)


@router.post("/api/trigger-summary/{user_id}")
async def trigger_summary(user_id: str):
    return await memory.trigger_summary(user_id)


@router.get("/api/get-home-data/{user_id}")
async def get_home_data(user_id: str):
    return await memory.get_home_data(user_id)


@router.get("/api/get-events/{user_id}")
async def get_events(user_id: str):
    return await memory.get_events(user_id)


@router.get("/api/get-mood/{user_id}")
async def get_mood(user_id: str):
    return await memory.get_mood(user_id)


@router.get("/api/members")
@router.get("/api/members_voice")
async def list_members():
    """List all users who have enrolled their voice."""
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


@router.delete("/api/voice-profile/{user_id}")
async def delete_voice_profile(user_id: str):
    member_voice_dir = "member_voice"
    file_path = os.path.join(member_voice_dir, f"{user_id}_profile.npy")
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "success", "message": f"Voice profile for '{user_id}' deleted."}
    else:
        raise HTTPException(status_code=404, detail="Voice profile not found")


# ── DEVICE STATUS & HEARTBEAT ─────────────────────────────────────

class DeviceStatusUpdate(BaseModel):
    bracelet: str = "Connected"
    dock: str = "Connected"

@router.post("/update")
async def update_device(data: DeviceStatusUpdate):
    """ESP32 calls this to push bracelet & dock status."""
    device_state["bracelet_last_seen"] = time.time()
    return {"status": "ok"}

@router.get("/device-status")
async def get_device_status():
    """App calls this to read latest ESP32 values."""
    is_alive = (time.time() - device_state["bracelet_last_seen"]) < 60
    return {
        "bracelet": "Connected" if is_alive else "Disconnected",
        "dock": "Connected"
    }


# ── SEEDING HELPERS ───────────────────────────────────────────────

class SeedMemoryRequest(BaseModel):
    user_id: str = "fish"
    entries: List[str] = [
        "I have a meeting with my professor tomorrow at 3 PM about the final project.",
        "I need to pick up groceries after class. We're out of milk and eggs.",
        "I'm feeling pretty stressed about the upcoming exam on Friday.",
        "My friend invited me to play basketball at the gym this Saturday at 10 AM.",
        "I should call mom tonight. It's been a while since we talked.",
    ]
    emotions: List[str] = ["Neutral", "Happy", "Angry", "Happy", "Sad"]

@router.post("/api/test-seed-memory")
async def test_seed_memory(req: SeedMemoryRequest):
    saved = 0
    for i, text in enumerate(req.entries):
        emotion = req.emotions[i] if i < len(req.emotions) else "Neutral"
        try:
            memory.save_memory(req.user_id, text, emotion, 0.95)
            saved += 1
        except Exception as e:
            logger.error(f"seed error: {e}")
    return {"status": "ok", "saved": saved, "user_id": req.user_id}


# ── Mount router AFTER all routes are registered ─────────────────
app.include_router(router)
