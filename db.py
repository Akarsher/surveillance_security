from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = "security_cam.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()

class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False)
    role = Column(String(32), nullable=False)  # 'authorized' | 'restricted'
    image_path = Column(String(512), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # np.float32 512-d tobytes()
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (UniqueConstraint("name", name="uq_person_name"),)

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    time = Column(DateTime, default=datetime.utcnow, index=True)
    reason = Column(Text, nullable=False)
    authorized = Column(Integer, default=0)
    restricted = Column(Integer, default=0)
    unknown = Column(Integer, default=0)
    snapshot_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(engine)