import cv2
import os
import csv
import face_recognition
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from datetime import datetime
import anyio
from fastapi.staticfiles import StaticFiles
from fastapi import Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil


app = FastAPI()

ADMIN_PASSWORD = "admin123"   # change this before final demo


# =============================
# PATHS
# =============================
KNOWN_FACES_DIR = "known_faces"
SNAPSHOT_DIR = "snapshots"
LOG_DIR = "logs"
EVENT_LOG = os.path.join(LOG_DIR, "events.csv")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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
known_encodings = []
known_names = []
known_roles = []

def load_known_faces():
    for role in ["authorized", "restricted"]:
        role_dir = os.path.join(KNOWN_FACES_DIR, role)
        if not os.path.exists(role_dir):
            continue

        for file in os.listdir(role_dir):
            path = os.path.join(role_dir, file)
            image = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(image)
            if enc:
                known_encodings.append(enc[0])
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

    process_every_n_frames = 3
    frame_count = 0
    last_locs, last_names, last_roles = [], [], []

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1

        if frame_count % process_every_n_frames == 0:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)

            last_locs, last_names, last_roles = [], [], []

            for (t, rgt, b, lft), enc in zip(locs, encs):
                matches = face_recognition.compare_faces(
                    known_encodings, enc, tolerance=0.6
                )

                name, role = "Unknown", "unknown"
                if True in matches:
                    i = matches.index(True)
                    name = known_names[i]
                    role = known_roles[i]

                last_locs.append((t*2, rgt*2, b*2, lft*2))
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
        # send the most recent alert immediately (if any)
        if last_alert:
            await websocket.send_json(last_alert)

        # keep the connection alive; we ignore client pings
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

# =============================
# ADMIN ROUTES  
# =============================

@app.get("/admin")
def admin_page():
    return FileResponse("admin.html")

@app.post("/admin/add_person")
async def add_person(
    password: str = Form(...),
    name: str = Form(...),
    role: str = Form(...),
    image: UploadFile = File(...)
):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin password")

    if role not in ["authorized", "restricted"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    save_dir = os.path.join(KNOWN_FACES_DIR, role)
    os.makedirs(save_dir, exist_ok=True)

    img_path = os.path.join(save_dir, f"{name}.jpg")

    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Load and encode face
    img = face_recognition.load_image_file(img_path)
    enc = face_recognition.face_encodings(img)

    if not enc:
        os.remove(img_path)
        raise HTTPException(status_code=400, detail="No face found in image")

    # Update in-memory lists (LIVE UPDATE)
    known_encodings.append(enc[0])
    known_names.append(name)
    known_roles.append(role)

    return {
        "status": "success",
        "message": f"{name} added as {role}"
    }
