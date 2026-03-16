from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn
from faster_whisper import WhisperModel
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import io
import glob
import os
import time
import asyncio
import json
import re
import chromadb
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.inference.classifiers import EncoderClassifier
from mem0 import Memory
import ollama

print("Setting up Memonic Cloud & Loading AI Models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. LOAD SPEECH & AUDIO MODELS
# ==========================================
model = None
verification_speaker_model = None
emotion_classifier = None

# ==========================================
# 2. GLOBALS: Memory & Profiles Cache
# ==========================================
memory = None
chroma_client = None
chroma_collection = None

# ==========================================
# 3. THE IN-MEMORY RAM CACHE
# ==========================================
profiles_cache = {}

def load_all_profiles():
    print("📂 Loading all enrolled voiceprints into RAM...")
    npy_files = glob.glob("*_profile.npy")
    for file_path in npy_files:
        user_id = file_path.replace("_profile.npy", "")
        enrolled_np = np.load(file_path)
        profiles_cache[user_id] = torch.from_numpy(enrolled_np).to(device)
        print(f"   -> Loaded profile for: {user_id}")

# ==========================================
# 4. SILENCE DETECTION + POPUP STORE
# ==========================================
last_audio_time: dict[str, float] = {}
popup_store: dict[str, str] = {}
session_buffer: dict[str, list[str]] = {}

SILENCE_THRESHOLD_SECONDS = 120
SILENCE_CHECK_INTERVAL = 30
MIN_WORDS_TO_SUMMARIZE = 10


async def build_summary(user_id: str, transcriptions: list[str]) -> str:
    joined = "\n".join(transcriptions)
    prompt = (
        f"You are a helpful personal assistant. "
        f"The user just finished a conversation. Here are the things they said:\n\n"
        f"{joined}\n\n"
        f"Write a short, warm, 1-2 sentence summary in the same language the user spoke. "
        f"End with a brief encouraging remark. "
        f"Do not use bullet points. Keep it under 40 words."
    )
    try:
        response = ollama.chat(
            model="llama3.2:8b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️  LLM summary failed: {e}")
        return f"Session recorded with {len(transcriptions)} entries."
#this function look weir

async def silence_watcher():
    print("👁️  Silence watcher started.")
    while True:
        await asyncio.sleep(SILENCE_CHECK_INTERVAL)
        now = time.time()
        for user_id in list(last_audio_time.keys()):
            elapsed = now - last_audio_time[user_id]
            if elapsed >= SILENCE_THRESHOLD_SECONDS:
                print(f"🔕 Silence detected for '{user_id}' ({elapsed:.0f}s) — triggering summary...")
                transcriptions = session_buffer.get(user_id, [])
                word_count = sum(len(t.split()) for t in transcriptions)
                if word_count >= MIN_WORDS_TO_SUMMARIZE:
                    summary = await build_summary(user_id, transcriptions)
                    popup_store[user_id] = summary
                    print(f"💬 Popup ready for '{user_id}': {summary}")
                else:
                    print(f"⏭️  Skipping — too few words ({word_count})")
                del last_audio_time[user_id]
                session_buffer.pop(user_id, None)


print(f"✅ System ready on: {device.upper()}")

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global model, verification_speaker_model, emotion_classifier, memory, chroma_client, chroma_collection

    print("📂 Loading models on startup...")

    model = WhisperModel("small", device=device, compute_type="int8")
    print("✓ Whisper loaded")

    verification_speaker_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    print("✓ Speaker Recognition loaded")

    emotion_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="pretrained_models/emotion-recognition-wav2vec2-IEMOCAP",
        run_opts={"device": device}
    )
    print("✓ Emotion Classifier loaded")

    print("📚 Initializing Mem0 Agentic Memory...")
    mem0_config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.2:3b",
                "temperature": 0.1,
                "max_tokens": 2000,
                "ollama_base_url": "http://localhost:11434"
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "ollama_base_url": "http://localhost:11434"
            }
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "mem0_collection", # 👈 Changed to avoid schema conflicts
                "path": "./mem0_memory"               # 👈 Changed path to fix the ValueError
            }
        }
    }
    memory = Memory.from_config(config_dict=mem0_config)
    print("✓ Mem0 Memory initialized")

    # This will now create cleanly without colliding with Mem0
    chroma_client = chromadb.PersistentClient(path="./memonic_memory")
    chroma_collection = chroma_client.get_or_create_collection("memonic_memory")
    print("✓ ChromaDB client initialized")
    print("✓ Mem0 Memory initialized")

    load_all_profiles()
    asyncio.create_task(silence_watcher())


# ==========================================
# 🟢 ENDPOINT 1: ENROLL A NEW VOICE
# ==========================================
@app.post("/api/enroll")
async def enroll_voice(user_id: str = Form(...), file: UploadFile = File(...)):
    print(f"📦 Enrolling new user: {user_id}")
    try:
        audio_bytes = await file.read()
        audio_io = io.BytesIO(audio_bytes)
        signal, fs = torchaudio.load(audio_io)

        # ✅ Duration check: must be 5–10 seconds
        duration_seconds = signal.shape[1] / fs
        if duration_seconds < 5.0:
            raise HTTPException(
                status_code=400,
                detail=f"Recording too short ({duration_seconds:.1f}s). Please record at least 5 seconds."
            )
        if duration_seconds > 12.0:  # a little buffer over 10s
            raise HTTPException(
                status_code=400,
                detail=f"Recording too long ({duration_seconds:.1f}s). Please keep it under 10 seconds."
            )

        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000).to(device)
            signal = resampler(signal.to(device))
        else:
            signal = signal.to(device)

        embedding = verification_speaker_model.encode_batch(signal).squeeze()
        embedding_np = embedding.cpu().numpy()
        np.save(f"{user_id}_profile.npy", embedding_np)
        profiles_cache[user_id] = embedding

        return {
            "status": "success",
            "message": f"User '{user_id}' enrolled.",
            "duration": round(duration_seconds, 2)
        }
    except HTTPException:
        raise  # re-raise your own validation errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 🔵 ENDPOINT 2: PROCESS AUDIO & SAVE TO MEMORY
# ==========================================
@app.post("/api/process-audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio_io = io.BytesIO(audio_bytes)
        signal, fs = torchaudio.load(audio_io)

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000).to(device)
            signal_device = resampler(signal.to(device))
        else:
            signal_device = signal.to(device)

        # --- A. SPEAKER IDENTIFICATION ---
        test_embedding = verification_speaker_model.encode_batch(signal_device).squeeze()
        best_match = "Unknown"
        best_score = 0.0

        for known_user, known_embedding in profiles_cache.items():
            similarity = F.cosine_similarity(
                known_embedding.unsqueeze(0),
                test_embedding.unsqueeze(0),
                dim=1
            )
            score = similarity.item()
            if score > best_score:
                best_score = score
                best_match = known_user

        identified_user = best_match if best_score > 0.75 else "Unknown"

        # --- B. EMOTION RECOGNITION ---
        try:
            emotion_signal = signal_device.squeeze(0).unsqueeze(0)
            out_prob, score, index, text_lab = emotion_classifier.classify_batch(emotion_signal)
            emotion_map = {'ang': 'Angry', 'hap': 'Happy', 'sad': 'Sad', 'neu': 'Neutral'}
            detected_emotion = emotion_map.get(text_lab[0], 'Unknown')
        except Exception as e:
            print(f"⚠️ Emotion detection failed: {e}")
            detected_emotion = "Unknown"

        # --- C. WHISPER TRANSCRIPTION ---
        signal_np = signal_device.squeeze().cpu().numpy()
        segments, info = model.transcribe(signal_np, beam_size=5, language="en")
        text = " ".join([seg.text for seg in list(segments)]).strip()

        if not text:
            text = "[No speech detected]"

        # --- D. SAVE DIRECTLY TO CHROMADB ---
        #fix if new user didn't have any user_member yet
        if text != "[No speech detected]" and identified_user != "Unknown":
            print(f"🧠 Saving to ChromaDB for user: {identified_user}...")
            import uuid
            res = ollama.embed(model="nomic-embed-text", input=text)
            chroma_collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[res["embeddings"][0]],
                documents=[text],
                metadatas=[{
                    "user_id": identified_user,
                    "memory": text,
                    "emotion": detected_emotion,
                    "speaker_confidence": best_score,
                    "timestamp": time.time()
                }]
            )
            print("💾 Memory saved successfully.")

            last_audio_time[identified_user] = time.time()
            if identified_user not in session_buffer:
                session_buffer[identified_user] = []
            session_buffer[identified_user].append(text)

        return {
            "status": "success",
            "identified_user": identified_user,
            "speaker_confidence": round(best_score, 4),
            "emotion": detected_emotion,
            "transcription": text
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 🟡 ENDPOINT 3: CHECK FOR PENDING POPUP
# ==========================================
@app.get("/api/check-popup/{user_id}")
async def check_popup(user_id: str):
    message = popup_store.get(user_id)
    if message:
        del popup_store[user_id]
        return {"has_popup": True, "message": message}
    return {"has_popup": False, "message": None}


# ==========================================
# 🔴 ENDPOINT 4: MANUAL TRIGGER (for testing)
# ==========================================
@app.post("/api/trigger-summary/{user_id}")
async def trigger_summary(user_id: str):
    transcriptions = session_buffer.get(user_id, [])
    if not transcriptions:
        return {"status": "no_session", "message": "No buffered transcriptions for this user."}

    summary = await build_summary(user_id, transcriptions)
    popup_store[user_id] = summary
    session_buffer.pop(user_id, None)
    last_audio_time.pop(user_id, None)

    return {"status": "triggered", "summary": summary}


# ==========================================
# 🟣 ENDPOINT 5: GET HOME DATA
# ==========================================
@app.get("/api/get-home-data/{user_id}")
#this function look weir also
async def get_home_data(user_id: str):
    try:
        results = chroma_collection.get(where={"user_id": user_id})
        documents = results.get("documents", [])

        if not documents:
            return {
                "highlights": "Start talking to Memonic to see your highlights.",
                "tasks": [],
                "updated_at": None
            }

        recent_text = "\n".join(documents[-5:])

        highlight_res = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": (
                f"Based on these memories:\n{recent_text}\n\n"
                f"Write ONE short highlight sentence (max 20 words). "
                f"Be warm and personal. No bullet points."
            )}]
        )
        highlights = highlight_res["message"]["content"].strip()

        task_res = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": (
                f"Based on these memories:\n{recent_text}\n\n"
                f"Extract 3-4 actionable tasks. "
                f"Reply ONLY with a JSON array of strings. "
                f'Example: ["Buy groceries", "Call Sarah"]'
            )}]
        )
        try:
            task_text = task_res["message"]["content"].strip()
            match = re.search(r'\[.*?\]', task_text, re.DOTALL)
            tasks = json.loads(match.group()) if match else ["Check your recent conversations"]
        except Exception:
            tasks = ["Check your recent conversations"]

        return {
            "highlights": highlights,
            "tasks": tasks,
            "updated_at": time.time()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)