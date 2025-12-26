import cv2
import os
import csv
import face_recognition
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from datetime import datetime

app = FastAPI()

# -----------------------------
# Paths
# -----------------------------
KNOWN_FACES_DIR = "known_faces"
SNAPSHOT_DIR = "snapshots"
LOG_DIR = "logs"
EVENT_LOG = os.path.join(LOG_DIR, "events.csv")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# Load known faces
# -----------------------------
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
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(file)[0])
                known_roles.append(role)

    print("[INFO] Known faces loaded")

load_known_faces()

# -----------------------------
# Webcam
# -----------------------------
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera.set(cv2.CAP_PROP_FPS, 10)

# -----------------------------
# Alert config
# -----------------------------
ALERT_DEBOUNCE_LIMIT = 3
violation_counter = 0
alert_active = False

# -----------------------------
# CSV Logger
# -----------------------------
def log_event(reason, auth_count, rest_count, unk_count, snapshot):
    file_exists = os.path.isfile(EVENT_LOG)

    with open(EVENT_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Reason",
                "Authorized",
                "Restricted",
                "Unknown",
                "Snapshot"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            reason,
            auth_count,
            rest_count,
            unk_count,
            snapshot
        ])

# -----------------------------
# MJPEG generator
# -----------------------------
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

            for (t, r, b, l), enc in zip(locs, encs):
                matches = face_recognition.compare_faces(
                    known_encodings, enc, tolerance=0.6
                )

                name, role = "Unknown", "unknown"
                if True in matches:
                    idx = matches.index(True)
                    name = known_names[idx]
                    role = known_roles[idx]

                last_locs.append((t*2, r*2, b*2, l*2))
                last_names.append(name)
                last_roles.append(role)

        # -----------------------------
        # Constraint logic
        # -----------------------------
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
            print("[ALERT]", reason)

        # -----------------------------
        # Draw boxes
        # -----------------------------
        for (t, rgt, b, lft), name, role in zip(
            last_locs, last_names, last_roles
        ):
            color = (0,255,0) if role=="authorized" else (0,165,255) if role=="restricted" else (0,0,255)
            cv2.rectangle(frame, (lft,t), (rgt,b), color, 2)
            cv2.putText(frame, f"{name} ({role})", (lft, t-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Status
        status_text = "SECURITY ALERT" if alert_active else "ACCESS OK"
        status_color = (0,0,255) if alert_active else (0,255,0)
        cv2.putText(frame, status_text, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

        ret, buf = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() +
            b"\r\n"
        )

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/stream")
def stream():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/events", response_class=HTMLResponse)
def events_page():
    rows = ""
    if os.path.exists(EVENT_LOG):
        with open(EVENT_LOG) as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                rows += f"""
                <tr>
                    <td>{r[0]}</td>
                    <td>{r[1]}</td>
                    <td>{r[2]}</td>
                    <td>{r[3]}</td>
                    <td>{r[4]}</td>
                    <td><a href='/{r[5]}' target='_blank'>View</a></td>
                </tr>
                """

    return f"""
    <html>
    <head>
        <title>Security Events</title>
        <style>
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}
            th {{ background-color: #333; color: white; }}
        </style>
    </head>
    <body>
        <h2>Security Event Log</h2>
        <table>
            <tr>
                <th>Time</th><th>Reason</th>
                <th>Auth</th><th>Restr</th><th>Unk</th><th>Snapshot</th>
            </tr>
            {rows}
        </table>
    </body>
    </html>
    """
