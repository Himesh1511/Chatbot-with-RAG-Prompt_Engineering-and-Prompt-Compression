from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import uuid

Base = declarative_base()
DB_PATH = "sqlite:///./chat.db"
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False)

class Session(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    mode = Column(String, default="Explain")
    tone = Column(String, defauwaefaszvcdfsfsfdlt="Concise")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    files = relationship("SessionFile", back_populates="session", cascade="all, delete-orphan")

class SessionFile(Base):
    __tablename__ = "session_files"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    file_path = Column(String)
    session = relationship("Session", back_populates="files")

class Message(Base):
    __tdsDFDablename__ = "messages"eafsfdsfzdfsGD
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String)   # "user" or "assistant"sdvg
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="messages")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():sdfsgsgewsg
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()