from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Employee(Base):
    __tablename__ = 'employees'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    emp_id = Column(String, unique=True, index=True)
    password = Column(String)
    mobile_number = Column(String)
    authorization = Column(Boolean, default=False)
    role = Column(String)
    photo_path = Column(String)

    def __repr__(self):
        return f"<Employee(username={self.username}, emp_id={self.emp_id}, role={self.role})>"

    @classmethod
    def create(cls, session, username, emp_id, password, mobile_number, authorization, role, photo_path):
        employee = cls(
            username=username,
            emp_id=emp_id,
            password=password,
            mobile_number=mobile_number,
            authorization=authorization,
            role=role,
            photo_path=photo_path
        )
        session.add(employee)
        session.commit()
        session.refresh(employee)
        return employee

    @classmethod
    def update(cls, session, emp_id, **kwargs):
        employee = session.query(cls).filter(cls.emp_id == emp_id).first()
        for key, value in kwargs.items():
            setattr(employee, key, value)
        session.commit()
        session.refresh(employee)
        return employee

    @classmethod
    def get_by_id(cls, session, emp_id):
        return session.query(cls).filter(cls.emp_id == emp_id).first()

    @classmethod
    def get_all(cls, session):
        return session.query(cls).all()