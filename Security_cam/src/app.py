import csv
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, Body, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from datetime import datetime, timedelta
import anyio
from fastapi.staticfiles import StaticFiles
import io
import shutil
import numpy as np
from typing import Optional
import threading
from sqlalchemy import select, delete, update
from sqlalchemy.exc import IntegrityError
from db import SessionLocal, init_db, Person, Event, Entry, Employee
from cloudflare_tunnel import cf_tunnel

app = FastAPI()

# ==============================
# PATHS
# ==============================
EMPLOYEE_DIR = "employees"
os.makedirs(EMPLOYEE_DIR, exist_ok=True)

# ==============================
# EMPLOYEE MANAGEMENT
# ==============================
@app.post("/admin/add_employee")
async def add_employee(
    username: str = Form(...),
    emp_id: str = Form(...),
    password: str = Form(...),
    mobile: str = Form(...),
    authorization: bool = Form(...),
    role: str = Form(...),
    photo: UploadFile = File(...)
):
    with SessionLocal() as s:
        existing_employee = s.scalar(select(Employee).where(Employee.username == username))
        if existing_employee:
            raise HTTPException(status_code=400, detail="Employee already exists")

        save_path = os.path.join(EMPLOYEE_DIR, f"{username}.jpg")
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)

        employee = Employee(
            username=username,
            emp_id=emp_id,
            password=password,
            mobile=mobile,
            authorization=authorization,
            role=role,
            photo_path=save_path
        )
        s.add(employee)
        s.commit()
    return {"status": "success", "message": f"Employee {username} added"}

@app.post("/admin/edit_employee")
async def edit_employee(
    emp_id: int = Form(...),
    username: str = Form(...),
    mobile: str = Form(...),
    authorization: bool = Form(...),
    role: str = Form(...),
    photo: Optional[UploadFile] = File(None)
):
    with SessionLocal() as s:
        employee = s.get(Employee, emp_id)
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")

        employee.username = username
        employee.mobile = mobile
        employee.authorization = authorization
        employee.role = role

        if photo:
            save_path = os.path.join(EMPLOYEE_DIR, f"{username}.jpg")
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(photo.file, buffer)
            employee.photo_path = save_path

        s.commit()
    return {"status": "success", "message": "Employee updated"}

@app.get("/admin/employees")
def get_employees():
    with SessionLocal() as s:
        employees = s.scalars(select(Employee)).all()
        return [{"username": emp.username, "emp_id": emp.emp_id, "mobile": emp.mobile, "authorization": emp.authorization, "role": emp.role, "photo": emp.photo_path} for emp in employees]

@app.post("/admin/delete_employee")
async def delete_employee(emp_id: int = Form(...)):
    with SessionLocal() as s:
        employee = s.get(Employee, emp_id)
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        os.remove(employee.photo_path)
        s.delete(employee)
        s.commit()
    return {"status": "success", "message": "Employee deleted"}

# ==============================
# ROUTES
# ==============================
@app.get("/admin/employees/view", response_class=HTMLResponse)
def employees_page():
    return HTMLResponse(content=open("templates/admin_employees.html").read())

# Other existing routes...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)