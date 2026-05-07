import os
import glob
import tempfile
import numpy as np
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, FastAPI, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
import logging

from core.database import SessionLocal
from core.models import Memory

logger = logging.getLogger(__name__)

router = APIRouter()
app = FastAPI()
app.include_router(router)

# Try loading models, but allow failure so the API still mounts
try:
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("base", device="cpu", compute_type="float32")
except Exception as e:
    logger.error(f"Failed to load whisper model: {e}")
    whisper_model = None

try:
    from speechbrain.inference.speaker import EncoderClassifier
    # Using the standard ecapa-tdnn model for embedding extraction
    speaker_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")
except Exception as e:
    logger.error(f"Failed to load speaker model: {e}")
    speaker_model = None

try:
    from transformers import pipeline
    emotion_classifier = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
except Exception as e:
    logger.error(f"Failed to load emotion model: {e}")
    emotion_classifier = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def identify_speaker(audio_path: str) -> str:
    if speaker_model is None:
        return "unknown"
    try:
        import torchaudio
        import torch
        signal, fs = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed (ECAPA-TDNN expects 16kHz)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
            
        embeddings = speaker_model.encode_batch(signal)
        emb_np = embeddings.squeeze().cpu().numpy()
        
        member_voice_dir = os.path.join(os.path.dirname(__file__), "member_voice")
        if not os.path.exists(member_voice_dir):
            return "unknown"
            
        best_match = "unknown"
        best_score = -1.0
        
        for npy_file in glob.glob(os.path.join(member_voice_dir, "*_profile.npy")):
            profile_emb = np.load(npy_file)
            # Cosine similarity
            score = np.dot(emb_np, profile_emb) / (np.linalg.norm(emb_np) * np.linalg.norm(profile_emb))
            if score > best_score and score > 0.5: # 0.5 is an example threshold
                best_score = score
                best_match = os.path.basename(npy_file).replace("_profile.npy", "")
                
        return best_match
    except Exception as e:
        logger.error(f"Speaker identification failed: {e}")
        return "unknown"

def get_emotion(audio_path: str) -> str:
    if emotion_classifier is None:
        return "unknown"
    try:
        result = emotion_classifier(audio_path)
        if result and len(result) > 0:
            return result[0]['label']
    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
    return "unknown"


from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, FastAPI, WebSocket, WebSocketDisconnect

@router.post("/audio")
async def process_audio(file: UploadFile = File(...), enroll_user: str = Form(None), db: Session = Depends(get_db)):
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
                import ai.models as ai_models
                import torchaudio
                signal, fs = torchaudio.load(temp_file.name)
                embedding = ai_models.encode_speaker(signal, fs)
                ai_models.save_profile(enroll_user, embedding)
                return {
                    "status": "ok",
                    "text": f"SUCCESS: Enrolled {enroll_user}",
                    "memory_saved": False
                }
            except Exception as e:
                logger.error(f"Enroll error: {e}")
                raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})
        
        # Step 2: Whisper -> transcript text
        transcript = ""
        try:
            if whisper_model:
                segments, info = whisper_model.transcribe(temp_file.name, beam_size=5)
                transcript = " ".join([segment.text for segment in segments]).strip()
        except Exception as e:
            logger.error(f"Whisper failed: {e}")

        # Step 3: ECAPA-TDNN -> return speaker name or "unknown"
        speaker = identify_speaker(temp_file.name)
        
        # Step 4: wav2vec2-based emotion detection -> emotion label
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
        
        # Step 6: Return JSON
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


@router.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    logger.info("WebSocket connected")
    try:
        while True:
            msg = await websocket.receive_text()
            if not msg.startswith("START") and not msg.startswith("ENROLL"):
                continue
                
            is_enroll = msg.startswith("ENROLL")
            enroll_user = msg.replace("ENROLL", "").strip() if is_enroll else None
            
            audio_data = bytearray()
            
            while True:
                message = await websocket.receive()
                if "bytes" in message:
                    audio_data.extend(message["bytes"])
                elif "text" in message:
                    if message["text"] == "STOP":
                        break
            
            if len(audio_data) < 1000:
                await websocket.send_text("ERROR: Audio too short")
                continue
                
            import wave
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            try:
                with wave.open(temp_file.name, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2) # 16-bit
                    wav_file.setframerate(16000)
                    wav_file.writeframes(audio_data)
                    
                if is_enroll and enroll_user:
                    try:
                        import ai.models as ai_models
                        import torchaudio
                        signal, fs = torchaudio.load(temp_file.name)
                        embedding = ai_models.encode_speaker(signal, fs)
                        ai_models.save_profile(enroll_user, embedding)
                        await websocket.send_text(f"SUCCESS: Enrolled {enroll_user}")
                    except Exception as e:
                        logger.error(f"Enroll error: {e}")
                        await websocket.send_text(f"ERROR: {e}")
                else:
                    transcript = ""
                    try:
                        if whisper_model:
                            segments, info = whisper_model.transcribe(temp_file.name, beam_size=5)
                            transcript = " ".join([segment.text for segment in segments]).strip()
                    except Exception as e:
                        logger.error(f"Whisper failed: {e}")

                    speaker = identify_speaker(temp_file.name)
                    emotion = get_emotion(temp_file.name)
                    
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
                    
                    # Store memory summary for popup
                    try:
                        import ai.memory as mem
                        import asyncio
                        # The popup requires user_id. For now, use speaker or default
                        user_id_for_popup = speaker if speaker != "unknown" else "user"
                        asyncio.create_task(mem.summarize_and_popup(user_id_for_popup, transcript, emotion))
                    except Exception as e:
                        logger.error(f"Popup error: {e}")
                    
                    await websocket.send_text(f"OK: {transcript} | Speaker: {speaker} | Emotion: {emotion}")
                    
            finally:
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(f"ERROR: {str(e)}")
        except:
            pass


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