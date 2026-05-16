import time
import asyncio
import json
import re
import uuid
import chromadb
import ollama
from datetime import datetime, timezone
from typing import Dict, List, Optional
try:
    from .config import SILENCE_CHECK_INTERVAL, SILENCE_THRESHOLD_SECONDS, MIN_WORDS_TO_SUMMARIZE, EMBEDDING_MODEL, LLM_MODEL_SUMMARY
except Exception:
    from config import SILENCE_CHECK_INTERVAL, SILENCE_THRESHOLD_SECONDS, MIN_WORDS_TO_SUMMARIZE, EMBEDDING_MODEL, LLM_MODEL_SUMMARY

# Simple in-memory cache for today_summary (avoids hammering ollama every poll)
_today_summary_cache: Dict[str, dict] = {}  # { user_id: { "summary": str, "cached_at": float, "meta": dict } }
TODAY_SUMMARY_TTL = 300  # re-generate every 5 minutes

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
        response = ollama.chat(model=LLM_MODEL_SUMMARY, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception:
        return f"Session recorded with {len(transcriptions)} entries."


def save_memory(user_id: str, text: str, emotion: str, speaker_confidence: float):
    global chroma_collection, session_buffer, last_audio_time, popup_store
    if chroma_collection is None:
        return
    # ollama 0.2+ uses embed(), 0.1.x uses embeddings() with different return shape
    if hasattr(ollama, 'embed'):
        res = ollama.embed(model=EMBEDDING_MODEL, input=text)
        embedding = res["embeddings"][0]
    else:
        res = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embedding = res["embedding"]
    chroma_collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[embedding],
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


async def summarize_and_popup(user_id: str, text: str, emotion: str) -> str:
    """
    Generate an immediate LLM summary for a single transcription from the ESP32
    and store it as a popup so the mobile app picks it up via check-popup polling.
    """
    prompt = (
        f"You are Memonic, a personal voice memory assistant. "
        f"The user '{user_id}' just said something. Their detected emotion is '{emotion}'.\n\n"
        f"What they said: \"{text}\"\n\n"
        f"Write a short, warm 1-2 sentence summary acknowledging what they said. "
        f"If the emotion is notable (Happy/Sad/Angry), briefly reflect it. "
        f"Keep under 30 words. No quotes."
    )
    try:
        response = ollama.chat(model=LLM_MODEL_SUMMARY, messages=[{"role": "user", "content": prompt}])
        summary = response["message"]["content"].strip()
    except Exception:
        summary = f"Heard from {user_id}: {text[:80]}"

    # Store in popup_store so the mobile app picks it up via /api/check-popup
    popup_store[user_id] = summary
    return summary


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
                "emotion": "Neutral",
                "indicator_position": 0.5,
                "distribution": {"Happy": 0, "Neutral": 0, "Sad": 0, "Angry": 0},
                "hourly_moods": [],
            }

        results = chroma_collection.get(where={"user_id": user_id})
        metadatas = results.get("metadatas", [])

        if not metadatas:
            return {
                "label": "No data yet",
                "sub": "Start talking to Memonic",
                "emotion": "Neutral",
                "indicator_position": 0.5,
                "distribution": {"Happy": 0, "Neutral": 0, "Sad": 0, "Angry": 0},
                "hourly_moods": [],
            }

        from collections import Counter, defaultdict

        # Pull emotions from today's entries only
        today_cutoff = time.time() - 86400
        recent_emotions = [
            m["emotion"] for m in metadatas
            if m.get("timestamp", 0) > today_cutoff and m.get("emotion") and m["emotion"] != "Unknown"
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
                "emotion": "Neutral",
                "indicator_position": 0.5,
                "distribution": {"Happy": 0, "Neutral": 0, "Sad": 0, "Angry": 0},
                "hourly_moods": [],
            }

        counts = Counter(recent_emotions)
        total = len(recent_emotions)
        distribution = {
            "Happy":   round(counts.get("Happy", 0)   / total * 100),
            "Neutral": round(counts.get("Neutral", 0) / total * 100),
            "Sad":     round(counts.get("Sad", 0)     / total * 100),
            "Angry":   round(counts.get("Angry", 0)   / total * 100),
        }
        dominant = counts.most_common(1)[0][0]

        position_map = {"Happy": 0.15, "Neutral": 0.50, "Sad": 0.70, "Angry": 0.92}
        indicator_position = position_map.get(dominant, 0.5)

        # ── Hourly mood breakdown (for the bar chart in home.js) ──────────
        emotion_to_val = {"Happy": 0.2, "Neutral": 0.5, "Sad": 0.7, "Angry": 0.9}
        hourly_buckets = defaultdict(list)
        now_ts = time.time()

        for m in metadatas:
            ts = m.get("timestamp", 0)
            emo = m.get("emotion", "")
            if ts > today_cutoff and emo and emo != "Unknown":
                hour = int((ts % 86400) / 3600)
                hourly_buckets[hour].append(emo)

        hourly_moods = []
        for hour in sorted(hourly_buckets.keys()):
            bucket = hourly_buckets[hour]
            dom_emo = Counter(bucket).most_common(1)[0][0]
            ampm = "a" if hour < 12 else "p"
            label = f"{hour % 12 or 12}{ampm}"
            hourly_moods.append({
                "time": label,
                "emotion": dom_emo,
                "val": emotion_to_val.get(dom_emo, 0.5),
            })

        if not hourly_moods:
            cur_hour = int((now_ts % 86400) / 3600)
            ampm = "a" if cur_hour < 12 else "p"
            hourly_moods = [{"time": f"{cur_hour % 12 or 12}{ampm}", "emotion": dominant, "val": emotion_to_val.get(dominant, 0.5)}]

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
        emoji_map = {"Happy": "😊", "Neutral": "😌", "Sad": "😔", "Angry": "😤"}

        return {
            "label": label_map.get(dominant, "Neutral"),
            "sub": sub_map.get(dominant, f"Based on {total} voice entries"),
            "emotion": dominant,
            "emoji": emoji_map.get(dominant, "😌"),
            "indicator_position": indicator_position,
            "distribution": distribution,
            "hourly_moods": hourly_moods,
        }

    except Exception as e:
        return {"error": str(e)}


async def today_summary(user_id: str, db_session=None) -> dict:
    """
    Generate a warm daily recap from today's Memory rows in SQLite.
    Cached for TODAY_SUMMARY_TTL seconds to avoid hammering ollama on every poll.
    Pass db_session=<SQLAlchemy Session> from the API caller.
    """
    cached = _today_summary_cache.get(user_id)
    if cached and (time.time() - cached.get("cached_at", 0)) < TODAY_SUMMARY_TTL:
        return cached

    fallback = {
        "summary": "Start talking to Memonic to see your daily recap.",
        "total_memories": 0,
        "speakers_seen": [],
        "dominant_emotion": "Neutral",
        "emoji": "😌",
        "time_range": None,
        "updated_at": time.time(),
    }

    if db_session is None:
        return fallback

    try:
        from core.models import Memory
        from datetime import date, time as dtime

        today_start = datetime.combine(date.today(), dtime.min)
        memories = (
            db_session.query(Memory)
            .filter(Memory.timestamp >= today_start)
            .order_by(Memory.timestamp.asc())
            .all()
        )

        if not memories:
            _today_summary_cache[user_id] = {**fallback, "cached_at": time.time()}
            return fallback

        transcripts = [m.transcript or "" for m in memories if m.transcript]
        speakers = list({m.speaker for m in memories if m.speaker and m.speaker.lower() not in ("unknown", "")})
        emotions = [m.emotion for m in memories if m.emotion and m.emotion != "Unknown"]
        first_ts = memories[0].timestamp
        last_ts = memories[-1].timestamp

        from collections import Counter
        dom_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "Neutral"
        emoji_map = {"Happy": "😊", "Neutral": "😌", "Sad": "😔", "Angry": "😤"}

        time_range = None
        try:
            time_range = f"{first_ts.strftime('%-I:%M %p')} - {last_ts.strftime('%-I:%M %p')}"
        except Exception:
            time_range = None

        joined = "\n".join(f"- {t}" for t in transcripts[-10:])
        prompt = (
            f"You are Memonic, a personal memory assistant. The user had {len(memories)} voice memories today.\n"
            f"Speakers: {', '.join(speakers) if speakers else 'unknown'}. Mood: {dom_emotion}.\n\n"
            f"Memories:\n{joined}\n\n"
            f"Write a warm, personal 2-sentence summary of what happened today. "
            f"Mention key themes or speakers. Under 40 words. No bullet points."
        )

        try:
            resp = ollama.chat(model=LLM_MODEL_SUMMARY, messages=[{"role": "user", "content": prompt}])
            summary_text = resp["message"]["content"].strip()
        except Exception:
            summary_text = (
                f"You had {len(memories)} voice memories today"
                + (f" with {', '.join(speakers)}." if speakers else ".")
            )

        result = {
            "summary": summary_text,
            "total_memories": len(memories),
            "speakers_seen": speakers,
            "dominant_emotion": dom_emotion,
            "emoji": emoji_map.get(dom_emotion, "😌"),
            "time_range": time_range,
            "updated_at": time.time(),
            "cached_at": time.time(),
        }
        _today_summary_cache[user_id] = result
        return result

    except Exception as e:
        return {**fallback, "error": str(e)}
