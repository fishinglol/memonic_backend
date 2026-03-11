from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
import models
from database import SessionLocal, engine, Base
from fastapi.middleware.cors import CORSMiddleware

# สร้าง Table ในฐานข้อมูล (memonic.db)
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # ในช่วงพัฒนาใส่ * เพื่อให้เข้าถึงได้ทุกที่
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency สำหรับดึง DB Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Models (สำหรับรับ-ส่งข้อมูล JSON) ---
class UserSchema(BaseModel):
    user_name: str
    password: str

class UserResponse(BaseModel):
    id: int
    user_name: str
    class Config:
        from_attributes = True

# --- Endpoints ---

# ใช้ POST สำหรับ Signin (สมัครสมาชิก)
@app.post("/signin", response_model=UserResponse)
def create_user(user: UserSchema, db: Session = Depends(get_db)):
    # 1. เช็คว่ามี User นี้อยู่หรือยัง
    exists = db.query(models.User).filter(models.User.user_name == user.user_name).first()
    if exists:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # 2. บันทึกลง Database
    new_user = models.User(user_name=user.user_name, password=user.password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# ใช้ POST สำหรับ Login (เพื่อความปลอดภัยของรหัสผ่าน)
@app.post("/login")
def login(user: UserSchema, db: Session = Depends(get_db)):
    # ดึงข้อมูลจาก DB มาเทียบ
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

@app.get("/sessions")
def get_history(db: Session = Depends(get_db)):
    history = db.query(models.ChatSession).all()
    return history