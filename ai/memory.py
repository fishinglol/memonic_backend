import time
import asyncio
import json
import re
import uuid
import chromadb
import ollama
from typing import Dict, List, Optional
try:
    from .config import SILENCE_CHECK_INTERVAL, SILENCE_THRESHOLD_SECONDS, MIN_WORDS_TO_SUMMARIZE
except Exception:
    from config import SILENCE_CHECK_INTERVAL, SILENCE_THRESHOLD_SECONDS, MIN_WORDS_TO_SUMMARIZE

# Runtime memory structures
chroma_client = None
chroma_collection = None
session_buffer: Dict[str, List[str]] = {}
popup_store: Dict[str, str] = {}
last_audio_time: Dict[str, float] = {}


async def init_memory(chroma_path: str = "./memonic_memory"):
    global chroma_client, chroma_collection
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = chroma_client.get_or_create_collection("memonic_memory")


async def build_summary(user_id: str, transcriptions: List[str]) -> str:
    joined = "\n".join(transcriptions)
    prompt = (
        f"You are a helpful personal assistant. The user just finished a conversation. "
        f"Here are the things they said:\n\n{joined}\n\n"
        f"Write a short, warm, 1-2 sentence summary in the same language the user spoke. "
        f"End with a brief encouraging remark. Keep under 40 words."
    )
    try:
        response = ollama.chat(model="llama3.2:8b", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception:
        return f"Session recorded with {len(transcriptions)} entries."


def save_memory(user_id: str, text: str, emotion: str, speaker_confidence: float):
    global chroma_collection, session_buffer, last_audio_time, popup_store
    if chroma_collection is None:
        return
    res = ollama.embed(model="nomic-embed-text", input=text)
    chroma_collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[res["embeddings"][0]],
        documents=[text],
        metadatas=[{
            "user_id": user_id,
            "memory": text,
            "emotion": emotion,
            "speaker_confidence": speaker_confidence,
            "timestamp": time.time()
        }]
    )
    last_audio_time[user_id] = time.time()
    session_buffer.setdefault(user_id, []).append(text)


async def silence_watcher():
    print("👁️  Silence watcher started.")
    while True:
        await asyncio.sleep(SILENCE_CHECK_INTERVAL)
        now = time.time()
        for user_id in list(last_audio_time.keys()):
            elapsed = now - last_audio_time[user_id]
            if elapsed >= SILENCE_THRESHOLD_SECONDS:
                transcriptions = session_buffer.get(user_id, [])
                word_count = sum(len(t.split()) for t in transcriptions)
                if word_count >= MIN_WORDS_TO_SUMMARIZE:
                    summary = await build_summary(user_id, transcriptions)
                    popup_store[user_id] = summary
                # clear
                last_audio_time.pop(user_id, None)
                session_buffer.pop(user_id, None)


def check_popup(user_id: str):
    message = popup_store.get(user_id)
    if message:
        del popup_store[user_id]
        return {"has_popup": True, "message": message}
    return {"has_popup": False, "message": None}


async def trigger_summary(user_id: str):
    transcriptions = session_buffer.get(user_id, [])
    if not transcriptions:
        return {"status": "no_session", "message": "No buffered transcriptions for this user."}
    summary = await build_summary(user_id, transcriptions)
    popup_store[user_id] = summary
    session_buffer.pop(user_id, None)
    last_audio_time.pop(user_id, None)
    return {"status": "triggered", "summary": summary}


async def get_home_data(user_id: str):
    try:
        if chroma_collection is None:
            return {"highlights": "Start talking to Memonic to see your highlights.", "tasks": [], "updated_at": None}
        results = chroma_collection.get(where={"user_id": user_id})
        documents = results.get("documents", [])
        if not documents:
            return {"highlights": "Start talking to Memonic to see your highlights.", "tasks": [], "updated_at": None}

        recent_text = "\n".join(documents[-5:])
        highlight_res = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": (
            f"Based on these memories:\n{recent_text}\n\nWrite ONE short highlight sentence (max 20 words). Be warm and personal. No bullet points.") }])
        highlights = highlight_res["message"]["content"].strip()

        task_res = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": (
            f"Based on these memories:\n{recent_text}\n\nExtract 3-4 actionable tasks. Reply ONLY with a JSON array of strings.") }])
        try:
            task_text = task_res["message"]["content"].strip()
            match = re.search(r'\[.*?\]', task_text, re.DOTALL)
            tasks = json.loads(match.group()) if match else ["Check your recent conversations"]
        except Exception:
            tasks = ["Check your recent conversations"]

        return {"highlights": highlights, "tasks": tasks, "updated_at": time.time()}
    except Exception as e:
        return {"error": str(e)}


async def get_events(user_id: str):
    try:
        if chroma_collection is None:
            return {"events": []}
        results = chroma_collection.get(where={"user_id": user_id})
        documents = results.get("documents", [])

        if not documents:
            return {"events": []}

        # Use last 10 memories as context
        recent_text = "\n".join(documents[-10:])

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": (
                f"Based on these personal memories:\n{recent_text}\n\n"
                f"Extract upcoming events, tasks, or appointments the user mentioned. "
                f"Reply ONLY with a valid JSON array. Each item must have exactly these fields:\n"
                f'  "title": short event name (max 4 words)\n'
                f'  "time": time string like "3:00 PM" or "Tomorrow" or "Friday"\n'
                f'  "location": place or empty string\n'
                f'  "icon": one of: calendar-outline, time-outline, people-outline, restaurant-outline, medkit-outline, barbell-outline, car-outline, home-outline\n'
                f"Return [] if no events found. No explanation, no markdown, just the JSON array."
            )}]
        )

        raw = response["message"]["content"].strip()
        # Strip markdown fences if model adds them
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        events = json.loads(match.group()) if match else []

        # Add stable IDs
        for i, e in enumerate(events):
            e["id"] = str(i)

        return {"events": events}

    except Exception as e:
        return {"error": str(e)}


async def get_mood(user_id: str):
    try:
        if chroma_collection is None:
            return {
                "label": "No data yet",
                "sub": "Start talking to Memonic",
                "emotion": "neu",
                "indicator_position": 0.5,
                "distribution": {"Happy": 0, "Neutral": 0, "Sad": 0, "Angry": 0}
            }
        
        results = chroma_collection.get(where={"user_id": user_id})
        metadatas = results.get("metadatas", [])

        if not metadatas:
            return {
                "label": "No data yet",
                "sub": "Start talking to Memonic",
                "emotion": "neu",
                "indicator_position": 0.5,
                "distribution": {"Happy": 0, "Neutral": 0, "Sad": 0, "Angry": 0}
            }

        # Pull emotions from today's entries only
        today = time.time() - 86400  # last 24 hours
        recent_emotions = [
            m["emotion"] for m in metadatas
            if m.get("timestamp", 0) > today and m.get("emotion") and m["emotion"] != "Unknown"
        ]

        # Fall back to last 20 if no entries today
        if not recent_emotions:
            recent_emotions = [
                m["emotion"] for m in metadatas[-20:]
                if m.get("emotion") and m["emotion"] != "Unknown"
            ]

        if not recent_emotions:
            return {
                "label": "No data yet",
                "sub": "Start talking to Memonic",
                "emotion": "neu",
                "indicator_position": 0.5,
                "distribution": {"Happy": 0, "Neutral": 0, "Sad": 0, "Angry": 0}
            }

        # Count distribution
        from collections import Counter
        counts = Counter(recent_emotions)
        total = len(recent_emotions)
        distribution = {
            "Happy":   round(counts.get("Happy", 0)   / total * 100),
            "Neutral": round(counts.get("Neutral", 0) / total * 100),
            "Sad":     round(counts.get("Sad", 0)     / total * 100),
            "Angry":   round(counts.get("Angry", 0)   / total * 100),
        }

        # Dominant emotion
        dominant = counts.most_common(1)[0][0]

        # Map emotion → indicator position (0.0 = calm left, 1.0 = stressed right)
        position_map = {
            "Happy":   0.15,
            "Neutral": 0.50,
            "Sad":     0.70,
            "Angry":   0.92,
        }
        indicator_position = position_map.get(dominant, 0.5)

        # Human-readable label
        label_map = {
            "Happy":   "Happy & Energized",
            "Neutral": "Calm & Focused",
            "Sad":     "Low Energy",
            "Angry":   "Tense & Stressed",
        }
        sub_map = {
            "Happy":   f"Feeling good — {counts.get('Happy', 0)} happy moments today",
            "Neutral": f"Steady — {total} voice entries analyzed",
            "Sad":     f"Take it easy — detected across {counts.get('Sad', 0)} entries",
            "Angry":   f"High stress — {counts.get('Angry', 0)} tense moments detected",
        }

        emoji_map = {
            "Happy":   "😊",
            "Neutral": "😌",
            "Sad":     "😔",
            "Angry":   "😤",
        }

        return {
            "label": label_map.get(dominant, "Neutral"),
            "sub": sub_map.get(dominant, f"Based on {total} voice entries"),
            "emotion": dominant,
            "emoji": emoji_map.get(dominant, "😌"),
            "indicator_position": indicator_position,
            "distribution": distribution,
        }

    except Exception as e:
        return {"error": str(e)}
