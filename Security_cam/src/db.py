from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./employees.db"  # Change this to your database URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    emp_id = Column(String, unique=True, index=True)
    password = Column(String)
    mobile_number = Column(String)
    authorization = Column(String)
    role = Column(String)
    photo = Column(LargeBinary)  # Store photo as binary data

def init_db():
    Base.metadata.create_all(bind=engine)