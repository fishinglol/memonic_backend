from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn
import io
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
    asyncio.create_task(memory.silence_watcher())


@app.post("/api/enroll")
async def enroll_voice(user_id: str = Form(...), file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        signal, fs = models.load_audio_bytes(audio_bytes)
        duration_seconds = signal.shape[1] / fs
        if duration_seconds < 5.0 or duration_seconds > 12.0:
            raise HTTPException(status_code=400, detail="Recording length invalid")

        embedding = models.encode_speaker(signal)
        models.save_profile(user_id, embedding)

        return {"status": "success", "message": f"User '{user_id}' enrolled.", "duration": round(duration_seconds,2)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process-audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        signal, fs = models.load_audio_bytes(audio_bytes)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)