import cv2
import os
import csv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from datetime import datetime, timedelta
import anyio
from fastapi.staticfiles import StaticFiles
from fastapi import Form, Body, UploadFile, File, HTTPException
import shutil
import numpy as np
from typing import Optional
import threading
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError
from db import SessionLocal, init_db, Person, Event

os.environ.setdefault("INSIGHTFACE_HOME", os.path.join(os.getcwd(), ".insightface"))

from insightface.app import FaceAnalysis

app = FastAPI()

# Init InsightFace (use env var for portability)
face_app = FaceAnalysis(name="buffalo_s", root=os.environ["INSIGHTFACE_HOME"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU mode

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
    with SessionLocal() as s:
        for role in ["authorized", "restricted"]:
            role_dir = os.path.join(KNOWN_FACES_DIR, role)
            if not os.path.exists(role_dir):
                continue
            for file in os.listdir(role_dir):
                name = os.path.splitext(file)[0]
                # skip if already in DB
                exists = s.scalar(select(Person).where(Person.name == name))
                if exists:
                    continue
                path = os.path.join(role_dir, file)
                img = cv2.imread(path)
                if img is None:
                    continue
                faces = face_app.get(img)
                if not faces:
                    continue
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                emb = face.normed_embedding
                if emb is None:
                    continue
                p = Person(
                    name=name, role=role, image_path=path, embedding=np_to_blob(emb)
                )
                s.add(p)
        s.commit()

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
# CAMERA
# =============================
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera.set(cv2.CAP_PROP_FPS, 10)

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
    # normalize to snapshots/<file>
    snap = snapshot
    if not snap.startswith("snapshots/"):
        snap = f"snapshots/{os.path.basename(snapshot)}"
    with SessionLocal() as s:
        s.add(Event(
            reason=reason, authorized=a, restricted=r, unknown=u,
            snapshot_path=snap
        ))
        # prune > 2 days using Event.time
        cutoff = datetime.utcnow() - timedelta(days=2)
        s.execute(delete(Event).where(Event.time < cutoff))
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
    last_reason = None  # track last fired reason

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1

        if frame_count % process_every_n_frames == 0:
            # detect + recognize (populate last_names/last_roles)
            last_locs, last_names, last_roles = [], [], []
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            faces = face_app.get(small)

            for f in faces:
                x1, y1, x2, y2 = (f.bbox * 2).astype(int)
                emb = f.normed_embedding
                name, role = "Unknown", "unknown"

                if emb is not None and enc_mat is not None:
                    sims = enc_mat @ emb.astype(np.float32)
                    i = int(np.argmax(sims)); best = float(sims[i])
                    if best >= 0.45:
                        name = known_names[i]
                        role = known_roles[i]

                last_locs.append((y1, x2, y2, x1))
                last_names.append(name)
                last_roles.append(role)

            # recompute counts every processed frame
            a = sum(1 for rr in last_roles if rr == "authorized")
            r = sum(1 for rr in last_roles if rr == "restricted")
            u = sum(1 for rr in last_roles if rr not in ("authorized", "restricted"))

            # prioritize violations: restricted > unknown > authorized count
            if r > 0:
                reason = "Restricted person detected"
            elif u > 0:
                reason = "Unknown person detected"
            elif a != 2:
                reason = f"Authorized count = {a}"
            else:
                reason = ""

            # update debounce
            if reason:
                violation_counter += 1
            else:
                violation_counter = 0
                alert_active = False
                last_reason = None

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
        for (t, rg, b, l), name, role in zip(last_locs, last_names, last_roles):
            color = (0,255,0) if role == "authorized" else (0,165,255) if role == "restricted" else (0,0,255)
            cv2.rectangle(frame, (l, t), (rg, b), color, 1)
            cv2.putText(frame, name, (l, max(0, t - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

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
def admin_events():
    items = []
    with SessionLocal() as s:
        events = s.scalars(select(Event).order_by(Event.time.desc())).all()
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

app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =============================
# ADMIN ROUTES  
# =============================

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

@app.get("/admin")
def admin_page():
    if not admin_logged_in:
        return FileResponse("admin_login.html")
    return FileResponse("admin_dashboard.html")

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
    data = [{"name": n, "role": r} for n, r in zip(known_names, known_roles)]
    return data

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

@app.post("/admin/update_image")
async def update_image(
    name: str = Form(...),
    role: str = Form(...),
    image: UploadFile = File(...)
):
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
        raise HTTPException(status_code=400, detail="No face found in image")
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.normed_embedding
    if emb is None:
        raise HTTPException(status_code=400, detail="Failed to compute embedding")

    with SessionLocal() as s:
        p = s.scalar(select(Person).where(Person.name == name))
        if p is None:
            # create if missing
            p = Person(name=name, role=role, image_path=img_path, embedding=np_to_blob(emb))
            s.add(p)
        else:
            p.role = role
            p.image_path = img_path
            p.embedding = np_to_blob(emb)
        s.commit()

    rebuild_cache()
    return {"status": "success", "message": f"Updated image for {name}"}
