import cv2
import os
import csv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from datetime import datetime
import anyio
from fastapi.staticfiles import StaticFiles
from fastapi import Form, Body, UploadFile, File, HTTPException
import shutil
import numpy as np
from typing import Optional

os.environ.setdefault("INSIGHTFACE_HOME", os.path.join(os.getcwd(), ".insightface"))

from insightface.app import FaceAnalysis

app = FastAPI()

# Init InsightFace
face_app = FaceAnalysis(name="buffalo_s", root="C:/Users/akars/.insightface")  # Use buffalo_l instead of buffalo_m
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
known_encodings = []  # ArcFace embeddings (np.array, shape [512])
known_names = []
known_roles = []

def load_known_faces():
    for role in ["authorized", "restricted"]:
        role_dir = os.path.join(KNOWN_FACES_DIR, role)
        if not os.path.exists(role_dir):
            continue
        for file in os.listdir(role_dir):
            path = os.path.join(role_dir, file)
            img = cv2.imread(path)
            if img is None:
                continue
            faces = face_app.get(img)
            if not faces:
                continue
            # Take the largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = face.normed_embedding  # 512-d normalized vector
            if emb is None:
                continue
            known_encodings.append(emb.astype(np.float32))
            known_names.append(os.path.splitext(file)[0])
            known_roles.append(role)

load_known_faces()

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
    exists = os.path.isfile(EVENT_LOG)
    with open(EVENT_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["Time", "Reason", "Auth", "Restr", "Unk", "Snapshot"])
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            reason, a, r, u, snapshot
        ])

# =============================
# MJPEG STREAM
# =============================
def generate_frames():
    global violation_counter, alert_active
    process_every_n_frames = 6  # increase skipping
    frame_count = 0
    last_locs, last_names, last_roles = [], [], []

    while True:
        success, frame = camera.read()
        if not success:
            break

        # downscale for detection (0.5x); keep original for display/snapshot
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        frame_count += 1

        if frame_count % process_every_n_frames == 0:
            last_locs, last_names, last_roles = [], [], []
            faces = face_app.get(small)  # detect on smaller frame
            for f in faces:
                # map bbox back to original coordinates
                x1, y1, x2, y2 = (f.bbox * 2).astype(int)
                emb = f.normed_embedding
                name, role = "Unknown", "unknown"

                if emb is not None and len(known_encodings) > 0:
                    sims = np.dot(np.stack(known_encodings), emb.astype(np.float32))
                    i = int(np.argmax(sims))
                    best = float(sims[i])

                    # slightly stricter threshold on CPU to reduce false positives
                    if best >= 0.45:
                        name = known_names[i]
                        role = known_roles[i]

                last_locs.append((y1, x2, y2, x1))  # (top, right, bottom, left)
                last_names.append(name)
                last_roles.append(role)

        # =============================
        # CONSTRAINT CHECK
        # =============================
        a = last_roles.count("authorized")
        r = last_roles.count("restricted")
        u = last_roles.count("unknown")

        reason = None
        if r > 0:
            reason = "Restricted person detected"
        elif u > 0:
            reason = "Unknown person detected"
        elif a != 2:
            reason = f"Authorized count = {a}"

        if reason:
            violation_counter += 1
        else:
            violation_counter = 0
            alert_active = False

        if violation_counter >= ALERT_DEBOUNCE_LIMIT and not alert_active:
            alert_active = True
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap = f"{SNAPSHOT_DIR}/alert_{ts}.jpg"
            cv2.imwrite(snap, frame)

            log_event(reason, a, r, u, snap)

            msg = {
                "time": ts,
                "reason": reason,
                "authorized": a,
                "restricted": r,
                "unknown": u,
                "snapshot": snap
            }
            # cache last alert for new clients
            global last_alert
            last_alert = msg

            # 🔴 PUSH REAL-TIME ALERT (safe from sync thread)
            anyio.from_thread.run(manager.broadcast, msg)

            print("[ALERT]", reason)

        # =============================
        # DRAW UI
        # =============================
        for (t, rgt, b, lft), name, role in zip(
            last_locs, last_names, last_roles
        ):
            color = (0,255,0) if role=="authorized" else (0,165,255) if role=="restricted" else (0,0,255)
            cv2.rectangle(frame, (lft,t), (rgt,b), color, 2)
            cv2.putText(frame, f"{name} ({role})", (lft,t-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        status = "SECURITY ALERT" if alert_active else "ACCESS OK"
        s_color = (0,0,255) if alert_active else (0,255,0)
        cv2.putText(frame, status, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, s_color, 3)

        ret, buf = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() +
            b"\r\n"
        )

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
    if os.path.exists(EVENT_LOG):
        with open(EVENT_LOG) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                rows += f"""
                <tr>
                    <td>{row[0]}</td><td>{row[1]}</td>
                    <td>{row[2]}</td><td>{row[3]}</td>
                    <td>{row[4]}</td>
                    <td><a href='/{row[5]}' target='_blank'>View</a></td>
                </tr>
                """

    return f"""
    <html>
    <head>
        <title>Security Dashboard</title>
        <script>
        const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(`${{scheme}}://${{location.host}}/ws/alerts`);
        ws.onopen = () => {{
            // keepalive ping so the server's receive loop stays satisfied
            ws.send('hello');
            setInterval(() => {{
                if (ws.readyState === WebSocket.OPEN) ws.send('ping');
            }}, 30000);
        }};
        ws.onmessage = function(event) {{
            const data = JSON.parse(event.data);
            alert(
                "🚨 SECURITY ALERT\\n" +
                data.reason + "\\n" +
                "Authorized: " + data.authorized
            );
        }};
        ws.onclose = () => console.log('ws closed');
        ws.onerror = (e) => console.error('ws error', e);
        </script>
    </head>
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

    known_encodings.append(emb.astype(np.float32))
    known_names.append(name)
    known_roles.append(role)

    return {"status": "success", "message": f"{name} added as {role}"}

@app.get("/admin/persons")
def admin_persons():
    # return current persons (name, role)
    items = [{"name": n, "role": r} for n, r in zip(known_names, known_roles)]
    return JSONResponse(items)

@app.post("/admin/delete_person")
async def delete_person(name: str = Form(...)):
    # find indices for this name
    indices = [i for i, n in enumerate(known_names) if n == name]
    if not indices:
        raise HTTPException(status_code=404, detail="Person not found")
    # remove files from both authorized/restricted if present
    removed_any = False
    for role in ["authorized", "restricted"]:
        path = os.path.join(KNOWN_FACES_DIR, role, f"{name}.jpg")
        if os.path.exists(path):
            os.remove(path)
            removed_any = True
    # remove from in-memory lists (reverse sort to pop safely)
    for i in sorted(indices, reverse=True):
        known_names.pop(i)
        known_roles.pop(i)
        known_encodings.pop(i)
    return {"status": "success", "message": f"Deleted {name}", "removed_file": removed_any}

@app.post("/admin/update_image")
async def update_image(
    name: str = Form(...),
    role: str = Form(...),
    image: UploadFile = File(...)
):
    if role not in ["authorized", "restricted"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    # save new image
    save_dir = os.path.join(KNOWN_FACES_DIR, role)
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, f"{name}.jpg")
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # compute new embedding
    img = cv2.imread(img_path)
    faces = face_app.get(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No face found in image")
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.normed_embedding
    if emb is None:
        raise HTTPException(status_code=400, detail="Failed to compute embedding")

    # update in-memory entry: replace or add
    replaced = False
    for i, (n, r) in enumerate(zip(known_names, known_roles)):
        if n == name:
            known_encodings[i] = emb.astype(np.float32)
            known_roles[i] = role
            replaced = True
            break
    if not replaced:
        known_names.append(name)
        known_roles.append(role)
        known_encodings.append(emb.astype(np.float32))

    return {"status": "success", "message": f"Updated image for {name}"}

@app.get("/admin/persons")
def get_persons():
    if not admin_logged_in:
        raise HTTPException(status_code=403)

    data = []
    for n, r in zip(known_names, known_roles):
        data.append({"name": n, "role": r})
    return data

@app.get("/admin/events")
def admin_events():
    items = []
    if os.path.exists(EVENT_LOG):
        with open(EVENT_LOG, newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if len(row) < 6:
                    continue
                snap = row[5]
                # normalize to snapshots/<file> so it works with /snapshots mount
                if not snap.startswith("snapshots/"):
                    # convert absolute or other paths to snapshots-relative if possible
                    try:
                        # keep only the filename and prepend snapshots/
                        filename = os.path.basename(snap)
                        snap = f"snapshots/{filename}"
                    except Exception:
                        pass
                items.append({
                    "time": row[0],
                    "reason": row[1],
                    "authorized": row[2],
                    "restricted": row[3],
                    "unknown": row[4],
                    "snapshot": snap,
                })
    return JSONResponse(items)
