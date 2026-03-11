from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String, unique=True, index=True)
    password = Column(String)

    
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)  # Groups messages into specific chats (e.g., "AES Encryption")
    role = Column(String)                    # Will be either 'user' or 'ai'
    content = Column(Text)                   # The actual message text
    timestamp = Column(DateTime(timezone=True), server_default=func.now()) # Auto-records the time