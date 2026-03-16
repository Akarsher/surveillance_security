import cv2
import os
import csv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from datetime import datetime, timedelta
import anyio
from fastapi.staticfiles import StaticFiles
from fastapi import Form, Body, UploadFile, File, HTTPException, Query
import io
import shutil
import numpy as np
from typing import Optional
import threading
from sqlalchemy import select, delete, update
from sqlalchemy.exc import IntegrityError
from db import SessionLocal, init_db, Person, Event, Entry
from cloudflare_tunnel import cf_tunnel
from utils import hash_password, verify_password, generate_emp_id

os.environ.setdefault("INSIGHTFACE_HOME", os.path.join(os.getcwd(), ".insightface"))

from insightface.app import FaceAnalysis
import onnxruntime as ort

app = FastAPI()

# Init InsightFace (use env var for portability)
available_providers = ort.get_available_providers()
use_cuda = "CUDAExecutionProvider" in available_providers
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
face_app = FaceAnalysis(name="buffalo_s", root=os.environ["INSIGHTFACE_HOME"], providers=providers)
face_app.prepare(ctx_id=0 if use_cuda else -1, det_size=(640, 640))

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

admin_logged_in = False

# =============================
# PATHS
# =============================
KNOWN_FACES_DIR = "known_faces"
SNAPSHOT_DIR = "snapshots"
LOG_DIR = "logs"
EVENT_LOG = os.path.join(LOG_DIR, "events.csv")
STATIC_DIR = "static"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/known_faces", StaticFiles(directory=KNOWN_FACES_DIR), name="known_faces")

# =============================
# WEBSOCKET MANAGER
# =============================
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# =============================
# LOAD KNOWN FACES
# =============================
# In-memory cache for fast matching
known_encodings = []  # list[np.ndarray float32, shape (512,)]
known_names = []
known_roles = []
enc_mat = None  # np.ndarray shape (N,512) for fast dot products

def np_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def blob_to_np(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

def rebuild_cache():
    global known_encodings, known_names, known_roles, enc_mat
    known_encodings, known_names, known_roles = [], [], []
    with SessionLocal() as s:
        people = s.scalars(select(Person)).all()
        for p in people:
            known_encodings.append(blob_to_np(p.embedding))
            known_names.append(p.name)
            known_roles.append(p.role)
    enc_mat = np.stack(known_encodings).astype(np.float32) if known_encodings else None

def cleanup_old_events():
    cutoff = datetime.utcnow() - timedelta(days=2)
    with SessionLocal() as s:
        # prune events older than 2 days based on Event.time
        s.execute(delete(Event).where(Event.time < cutoff))
        s.commit()

def load_known_faces_into_db_once():
    """Load faces from known_faces folder into DB (for backward compatibility)"""
    from utils import hash_password, generate_emp_id
    
    for role in ["authorized", "restricted"]:
        folder = os.path.join(KNOWN_FACES_DIR, role)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            continue
        
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            name = os.path.splitext(fname)[0]
            img_path = os.path.join(folder, fname)
            
            with SessionLocal() as s:
                # Check if already exists by name or image_path
                exists = s.query(Person).filter(
                    (Person.name == name) | (Person.image_path == img_path)
                ).first()
                
                if exists:
                    continue
                
                # Extract face embedding
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Could not read image: {img_path}")
                    continue
                
                faces = face_app.get(img)
                if not faces:
                    print(f"⚠️ No face found in: {img_path}")
                    continue
                
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                emb = face.normed_embedding
                if emb is None:
                    print(f"⚠️ Could not extract embedding from: {img_path}")
                    continue
                
                # Generate unique emp_id and username
                emp_id = generate_emp_id()
                # Make sure emp_id is unique
                while s.query(Person).filter(Person.emp_id == emp_id).first():
                    emp_id = generate_emp_id()
                
                # Generate username from name
                base_username = name.lower().replace(" ", "_").replace("-", "_")
                username = base_username
                counter = 1
                while s.query(Person).filter(Person.username == username).first():
                    username = f"{base_username}_{counter}"
                    counter += 1
                
                # Create person with all required fields
                person = Person(
                    name=name,
                    emp_id=emp_id,
                    username=username,
                    password=hash_password("password123"),  # Default password
                    mobile=None,
                    email=None,
                    role=role,
                    designation=None,
                    department=None,
                    image_path=img_path,
                    embedding=np_to_blob(emb),
                    is_active=1
                )
                
                s.add(person)
                s.commit()
                print(f"✅ Added {role} person: {name} (ID: {emp_id}, Username: {username})")
    
    rebuild_cache()

@app.on_event("startup")
def on_startup():
    init_db()
    load_known_faces_into_db_once()
    rebuild_cache()
    cleanup_old_events()
    # Optional: periodic cleanup thread (every 1 hour)
    def loop_cleanup():
        import time
        while True:
            try:
                cleanup_old_events()
            except Exception as e:
                print("Cleanup error:", e)
            time.sleep(3600)
    t = threading.Thread(target=loop_cleanup, daemon=True)
    t.start()

@app.on_event("shutdown")
def on_shutdown():
    try:
        camera.release()
    except Exception:
        pass

# =============================
# CAMERA CONFIGURATION
# =============================
CAMERA_CONFIG = {
    "type": None,  # Start with no camera selected
    "rtsp_url": "rtsp://rishitj:rishitj1972@192.168.1.2:554/stream2",
    "webcam_index": 0,
    "reconnect_delay": 5,
}

# =============================
# CAMERA CLASS WITH RECONNECTION
# =============================
import time

class CameraStream:
    def __init__(self, config: dict):
        self.config = config
        self.cap = None
        self.lock = threading.Lock()
        self.running = True
        self.active_type = None  # Track which camera is currently active
        self.last_error = None  # Track connection errors

    def _connect(self):
        with self.lock:
            self.last_error = None
            
            if self.cap is not None:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None

            if self.config["type"] is None:
                print("[CAMERA] No camera selected")
                self.active_type = None
                return False

            try:
                if self.config["type"] == "ip":
                    url = self.config["rtsp_url"]
                    print(f"[CAMERA] Connecting to IP camera...")
                    self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test connection by reading a frame
                    if not self.cap.isOpened():
                        self.last_error = "Failed to open RTSP stream. Check IP address and credentials."
                        print(f"[CAMERA] Error: {self.last_error}")
                        self.cap = None
                        self.active_type = None
                        return False
                    
                    # Try to read a test frame with timeout
                    ret, _ = self.cap.read()
                    if not ret:
                        self.last_error = "Connected but cannot read frames. Camera may be offline or credentials invalid."
                        print(f"[CAMERA] Error: {self.last_error}")
                        self.cap.release()
                        self.cap = None
                        self.active_type = None
                        return False
                        
                else:
                    print(f"[CAMERA] Connecting to webcam...")
                    self.cap = cv2.VideoCapture(self.config["webcam_index"])
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if not self.cap.isOpened():
                        self.last_error = "Failed to open webcam. It may be in use by another application."
                        print(f"[CAMERA] Error: {self.last_error}")
                        self.cap = None
                        self.active_type = None
                        return False

                self.active_type = self.config["type"]
                print(f"[CAMERA] Connected successfully to {self.active_type}")
                return True
                
            except Exception as e:
                self.last_error = f"Connection error: {str(e)}"
                print(f"[CAMERA] Exception: {self.last_error}")
                if self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                self.cap = None
                self.active_type = None
                return False

    def switch_camera(self, camera_type: str):
        """Switch to a different camera type ('webcam', 'ip', or None)"""
        # First stop current camera completely
        self.stop_camera()
        
        # Set new camera type
        self.config["type"] = camera_type
        
        # Connect to new camera
        if camera_type:
            return self._connect()
        return True

    def stop_camera(self):
        """Stop the current camera completely"""
        with self.lock:
            print("[CAMERA] Stopping camera...")
            
            # Release the capture device
            if self.cap is not None:
                try:
                    self.cap.release()
                    print("[CAMERA] Camera released")
                except Exception as e:
                    print(f"[CAMERA] Error releasing: {e}")
                finally:
                    self.cap = None
            
            # Reset state
            self.config["type"] = None
            self.active_type = None
            self.last_error = None
            
        print("[CAMERA] Camera stopped successfully")
        return True

    def read(self):
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            
            # Clear buffer for IP camera
            if self.config["type"] == "ip":
                self.cap.grab()
            
            return self.cap.read()

    def reconnect(self):
        if self.config["type"] is None:
            return False
        print(f"[CAMERA] Reconnecting in {self.config['reconnect_delay']} seconds...")
        time.sleep(self.config["reconnect_delay"])
        return self._connect()

    def release(self):
        self.running = False
        self.stop_camera()

    def is_opened(self):
        with self.lock:
            return self.cap is not None and self.cap.isOpened()

    def get_status(self):
        with self.lock:
            return {
                "type": self.config["type"],
                "active_type": self.active_type,
                "connected": self.cap is not None and self.cap.isOpened(),
                "rtsp_url": self.config["rtsp_url"].split("@")[-1] if self.config["rtsp_url"] else None,
                "error": self.last_error,
            }

# Initialize camera (but don't connect yet)
camera = CameraStream(CAMERA_CONFIG)

# =============================
# ALERT CONFIG
# =============================
ALERT_DEBOUNCE_LIMIT = 3
violation_counter = 0
alert_active = False
last_alert = None  # <-- cache last alert for new WS clients

# =============================
# CSV LOGGER
# =============================
def log_event(reason, a, r, u, snapshot):
    # normalize to URL-friendly path under /snapshots
    fname = os.path.basename(snapshot)
    snap = f"snapshots/{fname}"
    with SessionLocal() as s:
        s.add(Event(
            time=datetime.now(),
            reason=reason, authorized=a, restricted=r, unknown=u,
            snapshot_path=snap
        ))
        cutoff = datetime.now() - timedelta(days=2)
        s.execute(delete(Event).where(Event.time < cutoff))
        s.commit()

# =============================
# ENTRY LOGGER (for correct access)
# =============================
def log_entry(count: int, names: list):
    """Log a correct entry when exactly 1 authorized person is detected"""
    names_str = ",".join(names)
    with SessionLocal() as s:
        s.add(Entry(
            time=datetime.now(),
            count=count,
            names=names_str
        ))
        # Also prune old entries (older than 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        s.execute(delete(Entry).where(Entry.time < cutoff))
        s.commit()

# =============================
# MJPEG STREAM
# =============================
def generate_frames():
    global violation_counter, alert_active, last_alert
    process_every_n_frames = 5
    frame_count = 0
    last_locs, last_names, last_roles = [], [], []
    reason = ""
    a = r = u = 0
    last_reason = None
    consecutive_failures = 0
    last_entry_logged = None  # Track last entry to avoid duplicate logs

    while camera.running:
        # Check if camera is active
        if camera.config["type"] is None:
            # No camera selected - yield a placeholder frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No camera selected", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(placeholder, "Go to Dashboard to select a camera", (100, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(buffer) + b"\r\n"
            time.sleep(0.5)
            continue

        success, frame = camera.read()
        
        if not success or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= 30:
                camera.reconnect()
                consecutive_failures = 0
            time.sleep(0.1)
            continue
        
        consecutive_failures = 0
        frame_count += 1

        if frame_count % process_every_n_frames == 0:
            # detect + recognize (populate last_names/last_roles)
            last_locs, last_names, last_roles = [], [], []
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            faces = face_app.get(small)

            for f in faces:
                # bbox is [x1, y1, x2, y2], scale back to original frame
                bbox = (f.bbox * 2).astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                emb = f.normed_embedding
                name, role = "Unknown", "unknown"

                if emb is not None and enc_mat is not None:
                    sims = enc_mat @ emb.astype(np.float32)
                    i = int(np.argmax(sims))
                    best = float(sims[i])
                    if best >= 0.45:
                        name = known_names[i]
                        role = known_roles[i]

                # Store as (x1, y1, x2, y2) for clarity
                last_locs.append((x1, y1, x2, y2))
                last_names.append(name)
                last_roles.append(role)

            # recompute counts every processed frame
            a = sum(1 for rr in last_roles if rr == "authorized")
            r = sum(1 for rr in last_roles if rr == "restricted")
            u = sum(1 for rr in last_roles if rr not in ("authorized", "restricted"))

            # Get authorized names for entry logging
            auth_names = [n for n, ro in zip(last_names, last_roles) if ro == "authorized"]

            # prioritize violations: restricted > unknown > authorized count
            # CHANGED: Now requires exactly 1 authorized person (was 2)
            if r > 0:
                reason = "Restricted person detected"
            elif u > 0:
                reason = "Unknown person detected"
            elif a != 1:
                reason = f"Authorized count = {a}"
            else:
                reason = ""

            # update debounce
            if reason:
                violation_counter += 1
                last_entry_logged = None  # Reset entry tracking on violation
            else:
                violation_counter = 0
                alert_active = False
                last_reason = None
                
                # Log correct entry (exactly 1 authorized, no restricted/unknown)
                # Only log if names changed (avoid duplicate entries)
                current_entry_key = tuple(sorted(auth_names))
                if current_entry_key and current_entry_key != last_entry_logged:
                    log_entry(a, auth_names)
                    last_entry_logged = current_entry_key
                    print(f"[ENTRY] Logged: {auth_names}")

            # fire alert when:
            # - debounce reached AND
            # - either no active alert OR the violation type changed
            if violation_counter >= ALERT_DEBOUNCE_LIMIT and (not alert_active or reason != last_reason):
                alert_active = True
                last_reason = reason
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(SNAPSHOT_DIR, f"alert_{ts}.jpg")
                cv2.imwrite(snap_path, frame)

                log_event(reason, a, r, u, snap_path)

                msg = {
                    "time": ts,
                    "reason": reason,
                    "authorized": a,
                    "restricted": r,
                    "unknown": u,
                    "snapshot": f"snapshots/{os.path.basename(snap_path)}"
                }
                last_alert = msg
                anyio.from_thread.run(manager.broadcast, msg)
                print("[ALERT]", reason)

        # draw overlays (uses latest last_*)
        for (x1, y1, x2, y2), name, role in zip(last_locs, last_names, last_roles):
            color = (0, 255, 0) if role == "authorized" else (0, 165, 255) if role == "restricted" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Draw name above the box with better visibility
            label_y = max(y1 - 10, 20)
            cv2.putText(frame, name, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # yield MJPEG chunk
        ret, buffer = cv2.imencode('.jpg', frame)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(buffer) + b"\r\n"

# =============================
# ROUTES
# =============================
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/stream")
def stream():
    """Raw MJPEG stream for embedding in img tags"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Add a new route for the stream page with UI overlay
@app.get("/stream/view", response_class=HTMLResponse)
def stream_view():
    """Stream page with security camera UI overlay"""
    if not admin_logged_in:
        return FileResponse("templates/admin_login.html")
    return FileResponse("templates/stream.html")

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        if last_alert:
            await websocket.send_json(last_alert)  # send most recent on connect
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/events", response_class=HTMLResponse)
def events_page():
    rows = ""
    with SessionLocal() as s:
        events = s.scalars(select(Event).order_by(Event.time.desc())).all()
        for e in events:
            snap = e.snapshot_path or ""
            rows += f"""
                <tr>
                    <td>{e.time.strftime("%Y-%m-%d %H:%M:%S")}</td><td>{e.reason}</td>
                    <td>{e.authorized}</td><td>{e.restricted}</td>
                    <td>{e.unknown}</td>
                    <td><a href='/{snap}' target='_blank'>View</a></td>
                </tr>
            """
    return f"""
    <html>
    <head> ...same JS as before... </head>
    <body>
        <h2>Security Event Log</h2>
        <table border="1" cellpadding="6">
            <tr>
                <th>Time</th><th>Reason</th>
                <th>Auth</th><th>Restr</th><th>Unk</th><th>Snapshot</th>
            </tr>
            {rows}
        </table>
    </body>
    </html>
    """

@app.get("/admin/events")
def admin_events(start: Optional[str] = Query(None), end: Optional[str] = Query(None)):
    # start/end format: YYYY-MM-DD
    def parse_date(s: Optional[str]):
        try:
            return datetime.strptime(s, "%Y-%m-%d") if s else None
        except Exception:
            return None

    start_dt = parse_date(start)
    end_dt = parse_date(end)
    # end exclusive: next day for simpler query
    end_exclusive = end_dt + timedelta(days=1) if end_dt else None

    items = []
    with SessionLocal() as s:
        stmt = select(Event)
        if start_dt:
            stmt = stmt.where(Event.time >= start_dt)
        if end_exclusive:
            stmt = stmt.where(Event.time < end_exclusive)
        stmt = stmt.order_by(Event.time.desc())
        events = s.scalars(stmt).all()
        for e in events:
            items.append({
                "time": e.time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": e.reason,
                "authorized": e.authorized,
                "restricted": e.restricted,
                "unknown": e.unknown,
                "snapshot": e.snapshot_path or ""
            })
    return JSONResponse(items)

@app.get("/admin/events/export")
def admin_events_export(start: Optional[str] = Query(None), end: Optional[str] = Query(None)):
    # start/end format: YYYY-MM-DD
    def parse_date(s: Optional[str]):
        try:
            return datetime.strptime(s, "%Y-%m-%d") if s else None
        except Exception:
            return None

    start_dt = parse_date(start)
    end_dt = parse_date(end)
    end_exclusive = end_dt + timedelta(days=1) if end_dt else None

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Time", "Reason", "Authorized", "Restricted", "Unknown", "Snapshot"])

    with SessionLocal() as s:
        stmt = select(Event)
        if start_dt:
            stmt = stmt.where(Event.time >= start_dt)
        if end_exclusive:
            stmt = stmt.where(Event.time < end_exclusive)
        stmt = stmt.order_by(Event.time.desc())
        for e in s.scalars(stmt).all():
            writer.writerow([
                e.time.strftime("%Y-%m-%d %H:%M:%S"),
                e.reason,
                e.authorized,
                e.restricted,
                e.unknown,
                e.snapshot_path or ""
            ])

    output.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="events.csv"'}
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers=headers)

@app.get("/admin/entries")
def admin_entries(start: Optional[str] = Query(None), end: Optional[str] = Query(None)):
    """Get entry log (correct authorized accesses)"""
    def parse_date(s: Optional[str]):
        try:
            return datetime.strptime(s, "%Y-%m-%d") if s else None
        except Exception:
            return None

    start_dt = parse_date(start)
    end_dt = parse_date(end)
    end_exclusive = end_dt + timedelta(days=1) if end_dt else None

    items = []
    with SessionLocal() as s:
        stmt = select(Entry)
        if start_dt:
            stmt = stmt.where(Entry.time >= start_dt)
        if end_exclusive:
            stmt = stmt.where(Entry.time < end_exclusive)
        stmt = stmt.order_by(Entry.time.desc())
        entries = s.scalars(stmt).all()
        for e in entries:
            items.append({
                "date": e.time.strftime("%Y-%m-%d"),
                "time": e.time.strftime("%H:%M:%S"),
                "count": e.count,
                "names": e.names.split(",") if e.names else []
            })
    return JSONResponse(items)

@app.get("/admin/entries/export")
def admin_entries_export(start: Optional[str] = Query(None), end: Optional[str] = Query(None)):
    """Export entry log as CSV"""
    def parse_date(s: Optional[str]):
        try:
            return datetime.strptime(s, "%Y-%m-%d") if s else None
        except Exception:
            return None

    start_dt = parse_date(start)
    end_dt = parse_date(end)
    end_exclusive = end_dt + timedelta(days=1) if end_dt else None

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Time", "Count", "Names"])

    with SessionLocal() as s:
        stmt = select(Entry)
        if start_dt:
            stmt = stmt.where(Entry.time >= start_dt)
        if end_exclusive:
            stmt = stmt.where(Entry.time < end_exclusive)
        stmt = stmt.order_by(Entry.time.desc())
        for e in s.scalars(stmt).all():
            writer.writerow([
                e.time.strftime("%Y-%m-%d"),
                e.time.strftime("%H:%M:%S"),
                e.count,
                e.names
            ])

    output.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="entries.csv"'}
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers=headers)

@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard():
    if not admin_logged_in:
        return FileResponse("templates/admin_login.html")
    return FileResponse("templates/admin_dashboard.html")

@app.get("/admin/employees", response_class=HTMLResponse)
def admin_employees_page():
    if not admin_logged_in:
        return FileResponse("templates/admin_login.html")
    return FileResponse("templates/admin_employees.html")

@app.get("/admin/logs", response_class=HTMLResponse)
def admin_logs_page():
    if not admin_logged_in:
        return FileResponse("templates/admin_login.html")
    return FileResponse("templates/admin_logs.html")

@app.post("/admin/login")
def admin_login(data: dict = Body(...)):
    global admin_logged_in
    if (
        data.get("username") == ADMIN_USERNAME and
        data.get("password") == ADMIN_PASSWORD
    ):
        admin_logged_in = True
        return {"status": "success"}
    return {"status": "fail"}

@app.post("/admin/logout")
def admin_logout():
    """Logout admin and clear session"""
    global admin_logged_in
    admin_logged_in = False
    return {"status": "success", "message": "Admin logged out successfully"}

@app.post("/employee/logout")
def employee_logout():
    """Logout employee and clear session"""
    # No specific session handling for employees, just return success
    return {"status": "success", "message": "Employee logged out successfully"}

@app.post("/auth/admin/login")
def auth_admin_login(data: dict = Body(...)):
    """Admin login endpoint"""
    global admin_logged_in
    
    username = data.get("username", "").strip()
    password = data.get("password", "")
    
    if username == "admin" and password == "admin123":
        admin_logged_in = True
        return {
            "status": "success",
            "message": "Welcome, Admin!",
            "redirect": "/admin",  # Ensure this is included
            "user": {
                "username": "admin",
                "role": "admin"
            }
        }
    
    return JSONResponse(
        {"status": "fail", "message": "Invalid admin credentials"},
        status_code=401
    )

@app.post("/admin/add_person")
async def add_person(
    password: str = Form(...),
    name: str = Form(...),
    role: str = Form(...),
    image: UploadFile = File(...)
):
    if not admin_logged_in:
        raise HTTPException(status_code=403, detail="Not logged in")

    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin password")

    if role not in ["authorized", "restricted"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    save_dir = os.path.join(KNOWN_FACES_DIR, role)
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, f"{name}.jpg")
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    img = cv2.imread(img_path)
    faces = face_app.get(img)
    if not faces:
        os.remove(img_path)
        raise HTTPException(status_code=400, detail="No face found in image")
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.normed_embedding
    if emb is None:
        os.remove(img_path)
        raise HTTPException(status_code=400, detail="Failed to compute embedding")

    with SessionLocal() as s:
        p = Person(name=name, role=role, image_path=img_path, embedding=np_to_blob(emb))
        try:
            s.add(p)
            s.commit()
        except IntegrityError:
            s.rollback()
            raise HTTPException(status_code=409, detail="Person already exists")

    rebuild_cache()
    return {"status": "success", "message": f"{name} added as {role}"}

@app.get("/admin/persons")
def get_persons():
    if not admin_logged_in:
        raise HTTPException(status_code=403)
    with SessionLocal() as s:
        people = s.scalars(select(Person)).all()
        return [
            {
                "id": p.id,
                "name": p.name,
                "role": p.role,
                # Normalize path with forward slashes
                "image_path": p.image_path.replace("\\", "/") if p.image_path else None
            }
            for p in people
        ]

@app.post("/admin/edit_person")
async def edit_person(
    id: int = Form(...),
    name: str = Form(...),
    role: str = Form(...)
):
    if not admin_logged_in:
        raise HTTPException(status_code=403)
    if role not in ["authorized", "restricted"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    with SessionLocal() as s:
        p = s.get(Person, id)
        if not p:
            raise HTTPException(status_code=404, detail="Person not found")
        # rename file if name changed (optional)
        if name != p.name:
            new_path = os.path.join(KNOWN_FACES_DIR, role, f"{name}.jpg")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            try:
                if os.path.exists(p.image_path):
                    os.replace(p.image_path, new_path)
                p.image_path = new_path
            except Exception:
                # keep old path if move fails
                pass
        p.name = name
        p.role = role
        s.commit()
    rebuild_cache()
    return {"status": "success", "message": "Person updated"}

@app.post("/admin/update_image_by_id")
async def update_image_by_id(
    id: int = Form(...),
    role: str = Form(...),
    image: UploadFile = File(...)
):
    if not admin_logged_in:
        raise HTTPException(status_code=403)
    if role not in ["authorized", "restricted"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    with SessionLocal() as s:
        p = s.get(Person, id)
        if not p:
            raise HTTPException(status_code=404, detail="Person not found")

        save_dir = os.path.join(KNOWN_FACES_DIR, role)
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, f"{p.name}.jpg")
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        img = cv2.imread(img_path)
        faces = face_app.get(img)
        if not faces:
            raise HTTPException(status_code=400, detail="No face found in image")
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = face.normed_embedding
        if emb is None:
            raise HTTPException(status_code=400, detail="Failed to compute embedding")

        p.role = role
        p.image_path = img_path
        p.embedding = np_to_blob(emb)
        s.commit()

    rebuild_cache()
    return {"status": "success", "message": "Image updated"}

@app.post("/admin/delete_person")
async def delete_person(name: str = Form(...)):
    with SessionLocal() as s:
        people = s.scalars(select(Person).where(Person.name == name)).all()
        if not people:
            raise HTTPException(status_code=404, detail="Person not found")
        # remove files
        for p in people:
            try:
                if os.path.exists(p.image_path):
                    os.remove(p.image_path)
            except Exception:
                pass
            s.delete(p)
        s.commit()
    rebuild_cache()
    return {"status": "success", "message": f"Deleted {name}"}

@app.post("/admin/delete_person_by_id")
async def delete_person_by_id(id: int = Form(...)):
    if not admin_logged_in:
        raise HTTPException(status_code=403)
    with SessionLocal() as s:
        p = s.get(Person, id)
        if not p:
            raise HTTPException(status_code=404, detail="Person not found")
        try:
            if os.path.exists(p.image_path):
                os.remove(p.image_path)
        except Exception:
            pass
        s.delete(p)
        s.commit()
    rebuild_cache()
    return {"status": "success", "message": "Person deleted"}

# =============================
# CAMERA CONTROL ENDPOINTS
# =============================
@app.get("/admin/camera/status")
def camera_status():
    if not admin_logged_in:
        raise HTTPException(status_code=403)
    return camera.get_status()

@app.post("/admin/camera/switch")
def camera_switch(data: dict = Body(...)):
    if not admin_logged_in:
        raise HTTPException(status_code=403)
    
    camera_type = data.get("type")  # "webcam", "ip", or None
    
    if camera_type not in [None, "webcam", "ip"]:
        raise HTTPException(status_code=400, detail="Invalid camera type")
    
    success = camera.switch_camera(camera_type)
    status = camera.get_status()
    
    return {
        "status": "success" if success else "failed",
        "message": f"Connected to {camera_type}" if success else status.get("error", "Connection failed"),
        "connected": camera.is_opened(),
        "error": status.get("error")
    }

@app.post("/admin/camera/stop")
def camera_stop():
    if not admin_logged_in:
        raise HTTPException(status_code=403)
    
    camera.stop_camera()
    return {
        "status": "success", 
        "message": "Camera stopped",
        "connected": False
    }

# =============================
# CLOUDFLARE TUNNEL ENDPOINTS
# =============================

@app.get("/admin/tunnel/status")
def tunnel_status():
    """Get current tunnel status"""
    return JSONResponse(cf_tunnel.get_status())

@app.post("/admin/tunnel/start")
def start_tunnel():
    """Start Cloudflare tunnel"""
    if cf_tunnel.running:
        return JSONResponse({
            "status": "already_running",
            "url": cf_tunnel.public_url
        })
    
    url = cf_tunnel.start(port=8000)
    
    if url:
        return JSONResponse({
            "status": "started",
            "url": url
        })
    else:
        return JSONResponse({
            "status": "failed",
            "error": cf_tunnel.error or "Could not start tunnel"
        }, status_code=500)

@app.post("/admin/tunnel/stop")
def stop_tunnel():
    """Stop Cloudflare tunnel"""
    cf_tunnel.stop()
    return JSONResponse({"status": "stopped"})

@app.get("/admin/tunnel/history")
def tunnel_history():
    """Get URL history"""
    return JSONResponse(cf_tunnel.get_url_history())

# =============================
# EMPLOYEE MANAGEMENT ENDPOINTS
# =============================

# Serve the employees page
@app.get("/admin/employees", response_class=HTMLResponse)
def admin_employees_page():
    """Serve the employee management page"""
    if not admin_logged_in:
        return HTMLResponse('<script>window.location.href="/admin/login";</script>')
    return FileResponse("templates/admin_employees.html")


# Static routes FIRST (before dynamic routes)
@app.get("/admin/employees/list")
def list_employees():
    """Get all employees with full details"""
    if not admin_logged_in:
        raise HTTPException(status_code=403, detail="Not logged in")
    
    with SessionLocal() as s:
        employees = s.query(Person).order_by(Person.name).all()
        return [
            {
                "id": emp.id,
                "name": emp.name,
                "emp_id": emp.emp_id,
                "username": emp.username,
                "mobile": emp.mobile,
                "email": emp.email,
                "role": emp.role,
                "designation": emp.designation,
                "department": emp.department,
                "image_path": emp.image_path.replace("\\", "/") if emp.image_path else None,
                "is_active": emp.is_active if emp.is_active is not None else 1,
                "created_at": emp.created_at.strftime("%Y-%m-%d %H:%M") if emp.created_at else None,
                "updated_at": emp.updated_at.strftime("%Y-%m-%d %H:%M") if emp.updated_at else None
            }
            for emp in employees
        ]


@app.get("/admin/employee/generate-id")
def generate_employee_id():
    """Generate a unique employee ID"""
    if not admin_logged_in:
        raise HTTPException(status_code=403, detail="Not logged in")
    
    from utils import generate_emp_id
    
    with SessionLocal() as s:
        while True:
            new_id = generate_emp_id()
            existing = s.query(Person).filter(Person.emp_id == new_id).first()
            if not existing:
                return {"emp_id": new_id}


@app.get("/admin/departments")
def get_departments():
    """Get list of unique departments"""
    if not admin_logged_in:
        raise HTTPException(status_code=403)
    
    with SessionLocal() as s:
        departments = s.query(Person.department).distinct().all()
        return [d[0] for d in departments if d[0]]


@app.get("/admin/designations")
def get_designations():
    """Get list of unique designations"""
    if not admin_logged_in:
        raise HTTPException(status_code=403)
    
    with SessionLocal() as s:
        designations = s.query(Person.designation).distinct().all()
        return [d[0] for d in designations if d[0]]


@app.post("/admin/employee/add")
async def add_employee(
    name: str = Form(...),
    emp_id: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    mobile: str = Form(None),
    email: str = Form(None),
    role: str = Form("authorized"),
    designation: str = Form(None),
    department: str = Form(None),
    image: UploadFile = File(...)
):
    """Add a new employee with full details"""
    if not admin_logged_in:
        raise HTTPException(status_code=403, detail="Not logged in")

    if role not in ["authorized", "restricted"]:
        raise HTTPException(status_code=400, detail="Invalid role. Must be 'authorized' or 'restricted'")

    # Save image
    save_dir = os.path.join(KNOWN_FACES_DIR, role)
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, f"{emp_id}.jpg")
    
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Extract face embedding
    img = cv2.imread(img_path)
    faces = face_app.get(img)
    if not faces:
        os.remove(img_path)
        raise HTTPException(status_code=400, detail="No face found in image. Please upload a clear face photo.")
    
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.normed_embedding
    if emb is None:
        os.remove(img_path)
        raise HTTPException(status_code=400, detail="Failed to compute face embedding")

    # Hash password
    hashed_password = hash_password(password)

    # Save to database
    with SessionLocal() as s:
        # Check for duplicates
        existing_emp = s.query(Person).filter(Person.emp_id == emp_id).first()
        if existing_emp:
            os.remove(img_path)
            raise HTTPException(status_code=409, detail=f"Employee ID '{emp_id}' already exists")
        
        existing_user = s.query(Person).filter(Person.username == username).first()
        if existing_user:
            os.remove(img_path)
            raise HTTPException(status_code=409, detail=f"Username '{username}' already exists")

        employee = Person(
            name=name,
            emp_id=emp_id,
            username=username,
            password=hashed_password,
            mobile=mobile if mobile else None,
            email=email if email else None,
            role=role,
            designation=designation if designation else None,
            department=department if department else None,
            image_path=img_path,
            embedding=np_to_blob(emb),
            is_active=1
        )
        
        try:
            s.add(employee)
            s.commit()
            s.refresh(employee)
            emp_data = {
                "id": employee.id,
                "name": employee.name,
                "emp_id": employee.emp_id
            }
        except IntegrityError as e:
            s.rollback()
            os.remove(img_path)
            raise HTTPException(status_code=409, detail="Employee already exists")

    rebuild_cache()
    return {"status": "success", "message": f"Employee '{name}' added successfully", "employee": emp_data}


# Dynamic routes AFTER static routes
@app.get("/admin/employee/{emp_id}")
def get_employee(emp_id: str):
    """Get single employee details"""
    if not admin_logged_in:
        raise HTTPException(status_code=403)
    
    with SessionLocal() as s:
        emp = s.query(Person).filter(Person.emp_id == emp_id).first()
        if not emp:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        return {
            "id": emp.id,
            "name": emp.name,
            "emp_id": emp.emp_id,
            "username": emp.username,
            "mobile": emp.mobile,
            "email": emp.email,
            "role": emp.role,
            "designation": emp.designation,
            "department": emp.department,
            "image_path": emp.image_path.replace("\\", "/") if emp.image_path else None,
            "is_active": emp.is_active if emp.is_active is not None else 1,
            "created_at": emp.created_at.strftime("%Y-%m-%d %H:%M") if emp.created_at else None
        }


@app.put("/admin/employee/{emp_id}")
async def update_employee(
    emp_id: str,
    name: str = Form(None),
    new_emp_id: str = Form(None),
    role: str = Form(None),
    designation: str = Form(None),
    department: str = Form(None),
    mobile: str = Form(None),
    email: str = Form(None),
    is_active: int = Form(None),
    image: UploadFile = File(None)
):
    """Update employee details"""
    if not admin_logged_in:
        raise HTTPException(status_code=403, detail="Not logged in")

    if emp_id == "undefined" or not emp_id:
        raise HTTPException(status_code=400, detail="Invalid employee ID")

    with SessionLocal() as s:
        emp = s.query(Person).filter(Person.emp_id == emp_id).first()
        if not emp:
            raise HTTPException(status_code=404, detail="Employee not found")

        # Store name before session closes
        emp_name = emp.name

        # Update fields if provided
        if name is not None and name.strip():
            emp.name = name.strip()
            emp_name = name.strip()
        
        if new_emp_id is not None and new_emp_id.strip() and new_emp_id != emp_id:
            existing = s.query(Person).filter(Person.emp_id == new_emp_id).first()
            if existing:
                raise HTTPException(status_code=409, detail=f"Employee ID '{new_emp_id}' already exists")
            emp.emp_id = new_emp_id.strip()
        
        if role is not None and role in ["authorized", "restricted"]:
            emp.role = role
        
        if designation is not None:
            emp.designation = designation.strip() if designation.strip() else None
        
        if department is not None:
            emp.department = department.strip() if department.strip() else None
        
        if mobile is not None:
            emp.mobile = mobile.strip() if mobile.strip() else None
        
        if email is not None:
            emp.email = email.strip() if email.strip() else None
        
        if is_active is not None:
            emp.is_active = is_active

        # Update image if provided
        if image is not None and image.filename:
            save_dir = os.path.join(KNOWN_FACES_DIR, emp.role)
            os.makedirs(save_dir, exist_ok=True)
            img_path = os.path.join(save_dir, f"{emp.emp_id}.jpg")
            
            if emp.image_path and os.path.exists(emp.image_path) and emp.image_path != img_path:
                try:
                    os.remove(emp.image_path)
                except:
                    pass
            
            with open(img_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            img = cv2.imread(img_path)
            faces = face_app.get(img)
            if not faces:
                os.remove(img_path)
                raise HTTPException(status_code=400, detail="No face found in new image")
            
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = face.normed_embedding
            if emb is None:
                os.remove(img_path)
                raise HTTPException(status_code=400, detail="Failed to compute face embedding")

            emp.image_path = img_path
            emp.embedding = np_to_blob(emb)

        s.commit()

    rebuild_cache()
    return {"status": "success", "message": f"Employee '{emp_name}' updated successfully"}


@app.delete("/admin/employee/{emp_id}")
def delete_employee(emp_id: str):
    """Delete an employee"""
    if not admin_logged_in:
        raise HTTPException(status_code=403)
    
    if emp_id == "undefined" or not emp_id:
        raise HTTPException(status_code=400, detail="Invalid employee ID")
    
    with SessionLocal() as s:
        emp = s.query(Person).filter(Person.emp_id == emp_id).first()
        if not emp:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        # Remove image file
        if emp.image_path and os.path.exists(emp.image_path):
            try:
                os.remove(emp.image_path)
            except:
                pass
        
        s.delete(emp)
        s.commit()

    rebuild_cache()
    return {"status": "success", "message": "Employee deleted"}

# =============================
# EMPLOYEE SELF-SERVICE ENDPOINTS (for employee portal later)
# =============================

@app.post("/employee/login")
def employee_login(data: dict = Body(...)):
    """Employee login"""
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return JSONResponse({"status": "fail", "message": "Username and password required"}, status_code=400)
    
    with SessionLocal() as s:
        emp = s.query(Person).filter(Person.username == username).first()
        if not emp:
            return JSONResponse({"status": "fail", "message": "Invalid credentials"}, status_code=401)
        
        if not emp.is_active:
            return JSONResponse({"status": "fail", "message": "Account is deactivated"}, status_code=403)
        
        if not verify_password(password, emp.password):
            return JSONResponse({"status": "fail", "message": "Invalid credentials"}, status_code=401)
        
        return {
            "status": "success",
            "employee": {
                "id": emp.id,
                "name": emp.name,
                "emp_id": emp.emp_id,
                "username": emp.username,
                "designation": emp.designation,
                "department": emp.department
            }
        }


@app.put("/employee/change-password")
def employee_change_password(data: dict = Body(...)):
    """Employee change password"""
    username = data.get("username")
    current_password = data.get("current_password")
    new_password = data.get("new_password")
    
    if not all([username, current_password, new_password]):
        return JSONResponse({"status": "fail", "message": "All fields required"}, status_code=400)
    
    if len(new_password) < 6:
        return JSONResponse({"status": "fail", "message": "Password must be at least 6 characters"}, status_code=400)
    
    with SessionLocal() as s:
        emp = s.query(Person).filter(Person.username == username).first()
        if not emp:
            return JSONResponse({"status": "fail", "message": "User not found"}, status_code=404)
        
        if not verify_password(current_password, emp.password):
            return JSONResponse({"status": "fail", "message": "Current password is incorrect"}, status_code=401)
        
        emp.password = hash_password(new_password)
        s.commit()
    
    return {"status": "success", "message": "Password changed successfully"}


@app.put("/employee/update-profile")
async def employee_update_profile(
    username: str = Form(...),
    password: str = Form(...),
    mobile: str = Form(None),
    email: str = Form(None),
    image: UploadFile = File(None)
):
    """Employee update own profile (limited fields)"""
    with SessionLocal() as s:
        emp = s.query(Person).filter(Person.username == username).first()
        if not emp:
            return JSONResponse({"status": "fail", "message": "User not found"}, status_code=404)
        
        if not verify_password(password, emp.password):
            return JSONResponse({"status": "fail", "message": "Invalid password"}, status_code=401)
        
        # Update allowed fields
        if mobile is not None:
            emp.mobile = mobile
        
        if email is not None:
            emp.email = email
        
        # Update photo if provided
        if image is not None:
            save_dir = os.path.join(KNOWN_FACES_DIR, emp.role)
            os.makedirs(save_dir, exist_ok=True)
            img_path = os.path.join(save_dir, f"{emp.emp_id}.jpg")
            
            with open(img_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            img = cv2.imread(img_path)
            faces = face_app.get(img)
            if not faces:
                os.remove(img_path)
                return JSONResponse({"status": "fail", "message": "No face found in image"}, status_code=400)
            
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = face.normed_embedding
            if emb is None:
                os.remove(img_path)
                return JSONResponse({"status": "fail", "message": "Failed to process face"}, status_code=400)

            emp.image_path = img_path
            emp.embedding = np_to_blob(emb)
        
        s.commit()
    
    rebuild_cache()
    return {"status": "success", "message": "Profile updated successfully"}

@app.post("/employee/update-profile")
async def update_employee_profile(
    id: int = Form(...),
    name: str = Form(...),
    mobile: str = Form(...),
    password: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """Update employee profile"""
    with SessionLocal() as s:
        emp = s.query(Person).filter(Person.id == id).first()
        if not emp:
            raise HTTPException(status_code=404, detail="Employee not found")

        emp.name = name
        emp.mobile = mobile
        if password:
            emp.password = hash_password(password)

        # Handle profile picture upload
        if image:
            save_dir = os.path.join(KNOWN_FACES_DIR, emp.role)
            os.makedirs(save_dir, exist_ok=True)
            img_path = os.path.join(save_dir, f"{emp.emp_id}.jpg")
            
            # Save the new image
            with open(img_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            
            emp.image_path = img_path

        s.commit()
        return {
            "id": emp.id,
            "name": emp.name,
            "username": emp.username,
            "emp_id": emp.emp_id,
            "role": emp.role,
            "mobile": emp.mobile,
            "image_path": emp.image_path.replace("\\", "/") if emp.image_path else None
        }

@app.post("/employee/update-profile")
def update_employee_profile(data: dict = Body(...)):
    """Update employee profile"""
    emp_id = data.get("id")
    name = data.get("name")
    mobile = data.get("mobile")
    password = data.get("password")

    with SessionLocal() as s:
        emp = s.query(Person).filter(Person.id == emp_id).first()
        if not emp:
            raise HTTPException(status_code=404, detail="Employee not found")

        emp.name = name
        emp.mobile = mobile
        if password:
            emp.password = hash_password(password)

        s.commit()
        return {"status": "success", "message": "Profile updated successfully"}

# =============================
# AUTHENTICATION ENDPOINTS
# =============================

# Serve common login page
@app.get("/login", response_class=HTMLResponse)
def login_page():
    """Serve common login page"""
    return FileResponse("templates/login.html")


# Redirect root to login
@app.get("/", response_class=HTMLResponse)
def root():
    """Redirect to login"""
    return HTMLResponse('<script>window.location.href="/login";</script>')


# Admin login
@app.post("/auth/admin/login")
def auth_admin_login(data: dict = Body(...)):
    """Admin login endpoint"""
    global admin_logged_in
    
    username = data.get("username", "").strip()
    password = data.get("password", "")
    
    # Check admin credentials (you can change these)
    if username == "admin" and password == "admin123":
        admin_logged_in = True
        return {
            "status": "success",
            "message": "Welcome, Admin!",
            "redirect": "/admin",
            "user": {
                "username": "admin",
                "role": "admin"
            }
        }
    
    return JSONResponse(
        {"status": "fail", "message": "Invalid admin credentials"},
        status_code=401
    )


# Employee login
@app.post("/auth/employee/login")
def auth_employee_login(data: dict = Body(...)):
    """Employee login endpoint"""
    username = data.get("username", "").strip()
    password = data.get("password", "")
    
    if not username or not password:
        return JSONResponse(
            {"status": "fail", "message": "Username and password required"},
            status_code=400
        )
    
    with SessionLocal() as s:
        # Find by username or emp_id
        emp = s.query(Person).filter(
            (Person.username == username) | (Person.emp_id == username)
        ).first()
        
        if not emp:
            return JSONResponse(
                {"status": "fail", "message": "Invalid credentials"},
                status_code=401
            )
        
        if not emp.is_active:
            return JSONResponse(
                {"status": "fail", "message": "Account is deactivated. Contact admin."},
                status_code=403
            )
        
        if not verify_password(password, emp.password):
            return JSONResponse(
                {"status": "fail", "message": "Invalid credentials"},
                status_code=401
            )
        
        return {
            "status": "success",
            "message": f"Welcome, {emp.name}!",
            "redirect": "/employee",
            "user": {
                "id": emp.id,
                "name": emp.name,
                "emp_id": emp.emp_id,
                "username": emp.username,
                "designation": emp.designation,
                "department": emp.department,
                "image_path": emp.image_path.replace("\\", "/") if emp.image_path else None,
                "role": emp.role
            }
        }


# Face login
@app.post("/auth/face-login")
async def auth_face_login(image: UploadFile = File(...)):
    """Login using face recognition"""
    # Read image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(
            {"status": "fail", "message": "Invalid image"},
            status_code=400
        )
    
    # Detect faces
    faces = face_app.get(img)
    if not faces:
        return JSONResponse(
            {"status": "fail", "message": "No face detected. Please try again."},
            status_code=400
        )
    
    # Get largest face
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.normed_embedding
    
    if emb is None:
        return JSONResponse(
            {"status": "fail", "message": "Could not process face. Please try again."},
            status_code=400
        )
    
    # Match against known faces
    best_match = None
    best_score = 0.0
    threshold = 0.5
    
    with SessionLocal() as s:
        persons = s.query(Person).filter(Person.is_active == 1).all()
        
        for person in persons:
            if person.embedding:
                known_emb = blob_to_np(person.embedding)
                score = np.dot(emb, known_emb)
                
                if score > threshold and score > best_score:
                    best_score = score
                    best_match = person
        
        if best_match:
            return {
                "status": "success",
                "message": f"Welcome, {best_match.name}!",
                "redirect": "/employee",
                "user": {
                    "id": best_match.id,
                    "name": best_match.name,
                    "emp_id": best_match.emp_id,
                    "username": best_match.username,
                    "designation": best_match.designation,
                    "department": best_match.department,
                    "image_path": best_match.image_path.replace("\\", "/") if best_match.image_path else None,
                    "role": best_match.role
                },
                "confidence": float(best_score)
            }
    
    return JSONResponse(
        {"status": "fail", "message": "Face not recognized. Please use password login."},
        status_code=401
    )


# Serve employee dashboard
@app.get("/employee", response_class=HTMLResponse)
def employee_dashboard():
    """Serve employee dashboard"""
    return FileResponse("templates/employee_dashboard.html")


# =============================
# EMPLOYEE ATTENDANCE ENDPOINTS
# =============================

@app.get("/employee/today-status")
def employee_today_status(emp_id: str = Query(...)):
    """Get employee's today status"""
    from datetime import date
    
    today = date.today()
    today_start = datetime.combine(today, datetime.min.time())
    today_end = datetime.combine(today, datetime.max.time())
    
    with SessionLocal() as s:
        emp = s.query(Person).filter(Person.emp_id == emp_id).first()
        if not emp:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        # Get today's entries
        entries = s.query(Entry).filter(
            Entry.time >= today_start,
            Entry.time <= today_end,
            Entry.names.contains(emp.name)
        ).order_by(Entry.time).all()
        
        # Parse check-in/out times
        check_in_time = None
        check_out_time = None
        activities = []
        
        for entry in entries:
            entry_type = "check-in" if not check_in_time else "check-out"
            
            if not check_in_time:
                check_in_time = entry.time
            else:
                check_out_time = entry.time
            
            activities.append({
                "type": entry_type,
                "time": entry.time.strftime("%I:%M %p"),
                "method": "Face Recognition"
            })
        
        # Calculate hours
        hours_today = "0h 0m"
        if check_in_time:
            end_time = check_out_time or datetime.now()
            diff = end_time - check_in_time
            hours = int(diff.total_seconds() // 3600)
            minutes = int((diff.total_seconds() % 3600) // 60)
            hours_today = f"{hours}h {minutes}m"
        
        # Week stats (simplified)
        week_start = today - timedelta(days=today.weekday())
        week_entries = s.query(Entry).filter(
            Entry.time >= datetime.combine(week_start, datetime.min.time()),
            Entry.names.contains(emp.name)
        ).all()
        
        # Count unique days
        week_days = len(set(e.time.date() for e in week_entries))
        
        return {
            "status": "success",
            "checked_in": check_in_time is not None,
            "checked_out": check_out_time is not None,
            "check_in_time": check_in_time.strftime("%I:%M %p") if check_in_time else None,
            "check_out_time": check_out_time.strftime("%I:%M %p") if check_out_time else None,
            "hours_today": hours_today,
            "week_hours": f"{week_days * 8}h 0m",  # Simplified
            "week_days": week_days,
            "activities": activities
        }


@app.post("/employee/face-attendance")
async def employee_face_attendance(
    image: UploadFile = File(...),
    emp_id: str = Form(...),
    action: str = Form(...)  # 'in' or 'out'
):
    """Record attendance using face recognition"""
    # Read image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(
            {"status": "fail", "message": "Invalid image"},
            status_code=400
        )
    
    # Detect faces
    faces = face_app.get(img)
    if not faces:
        return JSONResponse(
            {"status": "fail", "message": "No face detected"},
            status_code=400
        )
    
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.normed_embedding
    
    if emb is None:
        return JSONResponse(
            {"status": "fail", "message": "Could not process face"},
            status_code=400
        )
    
    # Verify against employee's registered face
    with SessionLocal() as s:
        emp = s.query(Person).filter(Person.emp_id == emp_id).first()
        if not emp:
            return JSONResponse(
                {"status": "fail", "message": "Employee not found"},
                status_code=404
            )
        
        if not emp.embedding:
            return JSONResponse(
                {"status": "fail", "message": "No registered face found. Contact admin."},
                status_code=400
            )
        
        # Compare faces
        known_emb = blob_to_np(emp.embedding)
        score = np.dot(emb, known_emb)
        
        if score < 0.5:
            return JSONResponse(
                {"status": "fail", "message": "Face verification failed. Please try again."},
                status_code=401
            )
        
        # Record attendance
        now = datetime.now()
        
        # Save snapshot
        snapshot_dir = "snapshots/attendance"
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, f"{emp_id}_{action}_{now.strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(snapshot_path, img)
        
        # Create entry
        entry = Entry(
            time=now,
            count=1,
            names=emp.name
        )
        s.add(entry)
        s.commit()
        
        action_text = "Check-in" if action == "in" else "Check-out"
        return {
            "status": "success",
            "message": f"{action_text} successful at {now.strftime('%I:%M %p')}",
            "time": now.strftime("%I:%M %p"),
            "confidence": float(score)
        }

@app.get("/employee/logs")
def get_employee_logs(employee_id: int, start: Optional[str] = Query(None), end: Optional[str] = Query(None)):
    """Fetch logs specific to an employee"""
    def parse_date(s: Optional[str]):
        try:
            return datetime.strptime(s, "%Y-%m-%d") if s else None
        except Exception:
            return None

    start_dt = parse_date(start)
    end_dt = parse_date(end)
    end_exclusive = end_dt + timedelta(days=1) if end_dt else None

    with SessionLocal() as s:
        stmt = select(Entry).where(Entry.names.like(f"%{employee_id}%"))
        if start_dt:
            stmt = stmt.where(Entry.time >= start_dt)
        if end_exclusive:
            stmt = stmt.where(Entry.time < end_exclusive)
        stmt = stmt.order_by(Entry.time.desc())
        entries = s.scalars(stmt).all()

        return [
            {
                "date": e.time.strftime("%Y-%m-%d"),
                "time": e.time.strftime("%H:%M:%S"),
                "count": e.count,
                "names": e.names.split(",") if e.names else []
            }
            for e in entries
        ]

# Redirect old admin login to new common login
@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_redirect():
    """Redirect to common login page"""
    return HTMLResponse('<script>window.location.href="/login";</script>')

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Camera System")
    parser.add_argument("--tunnel", action="store_true", help="Start with Cloudflare tunnel enabled")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    args = parser.parse_args()
    
    print("")
    print("=" * 50)
    print("🔐 SECURITY CAMERA SYSTEM")
    print("=" * 50)
    
    # Start tunnel if requested
    if args.tunnel:
        print("")
        print("🚀 Starting with Cloudflare tunnel...")
        cf_tunnel.start(args.port)
    else:
        print("")
        print(f"📍 Running locally: http://localhost:{args.port}")
        print("💡 Tip: Run with --tunnel flag to enable remote access")
        print("   Example: python app.py --tunnel")
    
    print("")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        if cf_tunnel.running:
            cf_tunnel.stop()
