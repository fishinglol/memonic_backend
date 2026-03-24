from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func as sql_func
from pydantic import BaseModel
from typing import Optional, List
import models
from database import SessionLocal, engine, Base
from fastapi.middleware.cors import CORSMiddleware
import requests
import uuid

# สร้าง Table ในฐานข้อมูล (memonic.db)
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Ollama Configuration ---
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

# Dependency สำหรับดึง DB Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Ollama Helper ---
def call_ollama(messages: list) -> str:
    """Call Ollama llama3.2:3b and return the AI response text."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
        }, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "Sorry, I couldn't generate a response.")
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to Ollama. Please make sure it is running (ollama serve)."
    except Exception as e:
        return f"⚠️ Error from AI: {str(e)}"


# --- Pydantic Models ---
class UserSchema(BaseModel):
    user_name: str
    password: str

class UserResponse(BaseModel):
    id: int
    user_name: str
    class Config:
        from_attributes = True

class ChangePasswordRequest(BaseModel):
    user_name: str
    current_password: str
    new_password: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatMessageOut(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    class Config:
        from_attributes = True

class SessionOut(BaseModel):
    session_id: str
    title: str
    created_at: Optional[str] = None


# --- Auth Endpoints ---

@app.post("/signin", response_model=UserResponse)
def create_user(user: UserSchema, db: Session = Depends(get_db)):
    exists = db.query(models.User).filter(models.User.user_name == user.user_name).first()
    if exists:
        raise HTTPException(status_code=400, detail="User already exists")
    new_user = models.User(user_name=user.user_name, password=user.password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/login")
def login(user: UserSchema, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(
        models.User.user_name == user.user_name,
        models.User.password == user.password
    ).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    return {"message": "Login successful", "user_id": db_user.id}


@app.put("/api/change-password")
def change_password(req: ChangePasswordRequest, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(
        models.User.user_name == req.user_name,
        models.User.password == req.current_password
    ).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid current password or username"
        )
    
    db_user.password = req.new_password
    db.commit()
    return {"message": "Password updated successfully"}


@app.delete("/api/account/{user_name}")
def delete_account(user_name: str, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.user_name == user_name).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Optional: Delete chat records for this user. 
    # Current chat schema groups by session_id, we might not have user_id linked directly in ChatMessage yet,
    # but we can just delete the user.
    db.delete(db_user)
    db.commit()
    return {"message": "Account deleted successfully"}


# --- Chat Endpoints ---

@app.get("/api/members")
def list_members():
    import os, glob
    # Look into the AI folder's member_voice directory
    member_voice_dir = os.path.join(os.path.dirname(__file__), "..", "ai", "member_voice")
    if not os.path.exists(member_voice_dir):
        # Also try the root member_voice if it exists
        member_voice_dir = os.path.join(os.path.dirname(__file__), "..", "member_voice")
        if not os.path.exists(member_voice_dir):
            return []
    
    npy_files = glob.glob(os.path.join(member_voice_dir, "*_profile.npy"))
    members = []
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        user_id = filename.replace("_profile.npy", "")
        members.append(user_id)
    
    return sorted(members)


@app.post("/api/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """Send a message, get an AI reply via Ollama, and persist both to the DB."""
    # Generate a new session if none provided
    sid = req.session_id or str(uuid.uuid4())

    # Save user message
    user_msg = models.ChatMessage(session_id=sid, role="user", content=req.message)
    db.add(user_msg)
    db.commit()

    # Build conversation history for Ollama (load all prior messages in this session)
    history = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == sid)
        .order_by(models.ChatMessage.timestamp)
        .all()
    )
    ollama_messages = [
        {"role": "assistant" if m.role == "ai" else m.role, "content": m.content}
        for m in history
    ]

    # Call Ollama
    ai_text = call_ollama(ollama_messages)

    # Save AI response
    ai_msg = models.ChatMessage(session_id=sid, role="ai", content=ai_text)
    db.add(ai_msg)
    db.commit()

    # Derive title from first user message in this session
    first_msg = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == sid, models.ChatMessage.role == "user")
        .order_by(models.ChatMessage.timestamp)
        .first()
    )
    title = (first_msg.content[:50] + "...") if first_msg and len(first_msg.content) > 50 else (first_msg.content if first_msg else "New Chat")

    return {"session_id": sid, "reply": ai_text, "title": title}


@app.get("/api/chat/sessions")
def get_chat_sessions(db: Session = Depends(get_db)):
    """List all chat sessions with their auto-generated titles."""
    # Get distinct session_ids
    session_ids = (
        db.query(models.ChatMessage.session_id)
        .distinct()
        .all()
    )

    sessions = []
    for (sid,) in session_ids:
        # Get first user message as the title
        first_msg = (
            db.query(models.ChatMessage)
            .filter(models.ChatMessage.session_id == sid, models.ChatMessage.role == "user")
            .order_by(models.ChatMessage.timestamp)
            .first()
        )
        # Get the earliest timestamp
        earliest = (
            db.query(sql_func.min(models.ChatMessage.timestamp))
            .filter(models.ChatMessage.session_id == sid)
            .scalar()
        )

        title = "New Chat"
        if first_msg:
            title = (first_msg.content[:50] + "...") if len(first_msg.content) > 50 else first_msg.content

        sessions.append({
            "session_id": sid,
            "title": title,
            "created_at": str(earliest) if earliest else None,
        })

    # Sort newest first
    sessions.sort(key=lambda s: s["created_at"] or "", reverse=True)
    return sessions


@app.get("/api/chat/sessions/{session_id}")
def get_session_messages(session_id: str, db: Session = Depends(get_db)):
    """Get all messages for a specific chat session."""
    messages = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.timestamp)
        .all()
    )
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found")

    return [
        {
            "role": m.role,
            "content": m.content,
            "timestamp": str(m.timestamp) if m.timestamp else None,
        }
        for m in messages
    ]


@app.get("/sessions")
def get_history(db: Session = Depends(get_db)):
    history = db.query(models.ChatMessage).all()
    return history