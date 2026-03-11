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
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.inference.classifiers import EncoderClassifier
from mem0 import Memory

print("Setting up Memonic Cloud & Loading AI Models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. LOAD SPEECH & AUDIO MODELS
# ==========================================
# Whisper for Transcription (Quantized to int8 to save RAM)
model = WhisperModel("small", device=device, compute_type="int8")

# SpeechBrain for Speaker Verification
verification_speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

# SpeechBrain for Emotion Recognition
emotion_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion-recognition-wav2vec2-IEMOCAP",
    run_opts={"device": device} 
)

# ==========================================
# 2. SETUP MEM0 (LOCAL LLM + CHROMA DB)
# ==========================================
print("📚 Initializing Mem0 Agentic Memory...")
mem0_config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:1b", # Use a tiny model here for edge hardware!
            "temperature": 0.1,
            "base_url": "http://localhost:11434" # Default Ollama port
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "memonic_memory",
            "path": "./memonic_memory",
        }
    }
}
memory = Memory.from_config(config_dict=mem0_config)

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

load_all_profiles()

print(f"✅ System ready on: {device.upper()}")

app = FastAPI()

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
        embedding = verification_speaker_model.encode_batch(signal.to(device)).squeeze()
        
        # Save to Hard Drive
        embedding_np = embedding.cpu().numpy()
        np.save(f"{user_id}_profile.npy", embedding_np)
        
        # Add to RAM Cache
        profiles_cache[user_id] = embedding
        
        return {"status": "success", "message": f"User '{user_id}' enrolled and cached."}
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
        signal_device = signal.to(device)
        
        # --- A. SPEAKER IDENTIFICATION ---
        test_embedding = verification_speaker_model.encode_batch(signal_device).squeeze(0)
        
        best_match = "Unknown"
        best_score = 0.0
        
        for known_user, known_embedding in profiles_cache.items():
            similarity = F.cosine_similarity(known_embedding, test_embedding, dim=0)
            score = similarity.item()
            if score > best_score:
                best_score = score
                best_match = known_user
        
        identified_user = best_match if best_score > 0.75 else "Unknown"

        # --- B. EMOTION RECOGNITION ---
        out_prob, score, index, text_lab = emotion_classifier.classify_batch(signal_device)
        emotion_map = {'ang': 'Angry', 'hap': 'Happy', 'sad': 'Sad', 'neu': 'Neutral'}
        detected_emotion = emotion_map.get(text_lab[0], 'Unknown')
        
        # --- C. WHISPER TRANSCRIPTION ---
        audio_io.seek(0) 
       signal_np = signal.squeeze().cpu().numpy()
        segments, info = model.transcribe(signal_np, beam_size=5, language="en")
        
        if not text:
            text = "[No speech detected]"

        # --- D. SAVE TO MEM0 ---
        if text != "[No speech detected]" and identified_user != "Unknown":
            print(f"🧠 Sending facts to Mem0 for user: {identified_user}...")
            
            # This triggers Ollama to extract the context and save it to ChromaDB
            memory.add(
                messages=[{"role": "user", "content": text}],
                user_id=identified_user,
                metadata={
                    "emotion": detected_emotion,
                    "speaker_confidence": best_score
                }
            )
            print("💾 Memory saved successfully.")

        # --- E. RETURN JSON ---
        return {
            "status": "success",
            "identified_user": identified_user,
            "speaker_confidence": round(best_score, 4),
            "emotion": detected_emotion,
            "transcription": text
        }
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)