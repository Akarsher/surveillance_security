from sqlalchemy import create_engine, Column, Integer, String, DateTime, LargeBinary, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

Base = declarative_base()

class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Basic Info
    name = Column(String(100), nullable=False)
    emp_id = Column(String(50), unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    
    # Contact (optional)
    mobile = Column(String(20), nullable=True)
    email = Column(String(100), nullable=True)
    
    # Role & Authorization
    role = Column(String(20), default="authorized")
    designation = Column(String(100), nullable=True)
    department = Column(String(100), nullable=True)
    
    # Face Recognition
    image_path = Column(Text, nullable=True)
    embedding = Column(LargeBinary, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Integer, default=1)


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(DateTime)
    reason = Column(String(255))
    authorized = Column(Integer)
    restricted = Column(Integer)
    unknown = Column(Integer)
    snapshot_path = Column(Text)


class Entry(Base):
    __tablename__ = "entries"
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(DateTime)
    count = Column(Integer)
    names = Column(Text)


DATABASE_URL = "sqlite:///./security.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)