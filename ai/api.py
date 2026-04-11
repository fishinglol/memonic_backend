from fastapi import FastAPI, HTTPException, Request
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


@app.post("/api/esp32-audio")
async def esp32_audio(request: Request):
    """
    Receive raw WAV bytes from the ESP32 bracelet (no base64).
    Run full pipeline: speaker ID → emotion → transcription → LLM summary → save to memory.
    Triggers a popup for the mobile app via check-popup polling.
    """
    try:
        # Mark bracelet as alive whenever it sends audio
        device_state["bracelet_last_seen"] = time.time()

        audio_bytes = await request.body()
        if len(audio_bytes) < 44:
            raise HTTPException(status_code=400, detail="Audio too short (need WAV with header)")

        signal, fs = models.load_audio_bytes(audio_bytes, "audio.wav")
        if signal.shape[0] > 1:
            signal = models.to_mono(signal)

        # Speaker identification
        test_embedding = models.encode_speaker(signal)
        identified_user, best_score = models.match_speaker(test_embedding)

        # Emotion detection
        detected_emotion = models.classify_emotion(signal)

        # Transcription
        text = models.transcribe(signal)
        if not text:
            text = "[No speech detected]"

        summary = ""
        if text != "[No speech detected]" and identified_user != "Unknown":
            # Save to ChromaDB (feeds fetchHomeData, fetchEvents, fetchMood)
            memory.save_memory(identified_user, text, detected_emotion, best_score)
            # Generate immediate LLM summary and store popup (feeds popupInterval in mobile app)
            summary = await memory.summarize_and_popup(identified_user, text, detected_emotion)

        return {
            "identified_user": identified_user,
            "speaker_confidence": round(best_score, 4),
            "emotion": detected_emotion,
            "transcription": text,
            "summary": summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("❌ ERROR in esp32_audio:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)