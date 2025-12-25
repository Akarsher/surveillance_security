import cv2
import os
import face_recognition
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from datetime import datetime

app = FastAPI()

# -----------------------------
# Load known faces
# -----------------------------
KNOWN_FACES_DIR = "known_faces"
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

            if len(encodings) == 0:
                print(f"[WARNING] No face found in {path}")
                continue

            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
            known_roles.append(role)

    print("[INFO] Known faces loaded:")
    for n, r in zip(known_names, known_roles):
        print(f" - {n} ({r})")

load_known_faces()

# -----------------------------
# Webcam setup (LOW LATENCY)
# -----------------------------
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera.set(cv2.CAP_PROP_FPS, 10)

# -----------------------------
# Alert configuration
# -----------------------------
ALERT_DEBOUNCE_LIMIT = 3
violation_counter = 0
alert_active = False

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# -----------------------------
# MJPEG frame generator
# -----------------------------
def generate_frames():
    global violation_counter, alert_active

    process_every_n_frames = 3
    frame_count = 0

    last_face_locations = []
    last_face_names = []
    last_face_roles = []

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1

        # ----------------------------------
        # Run recognition periodically
        # ----------------------------------
        if frame_count % process_every_n_frames == 0:

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            last_face_locations = []
            last_face_names = []
            last_face_roles = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                matches = face_recognition.compare_faces(
                    known_encodings, face_encoding, tolerance=0.6
                )

                name = "Unknown"
                role = "unknown"

                if True in matches:
                    index = matches.index(True)
                    name = known_names[index]
                    role = known_roles[index]

                last_face_locations.append(
                    (top * 2, right * 2, bottom * 2, left * 2)
                )
                last_face_names.append(name)
                last_face_roles.append(role)

        # ----------------------------------
        # SECURITY CONSTRAINT CHECK
        # ----------------------------------
        authorized_count = last_face_roles.count("authorized")
        restricted_count = last_face_roles.count("restricted")
        unknown_count = last_face_roles.count("unknown")

        violation_reason = None

        if restricted_count > 0:
            violation_reason = "Restricted person detected"
        elif unknown_count > 0:
            violation_reason = "Unknown person detected"
        elif authorized_count != 2:
            violation_reason = f"Authorized count = {authorized_count} (expected 2)"

        if violation_reason:
            violation_counter += 1
        else:
            violation_counter = 0
            alert_active = False

        # ----------------------------------
        # Trigger alert after debounce
        # ----------------------------------
        if violation_counter >= ALERT_DEBOUNCE_LIMIT and not alert_active:
            alert_active = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(
                SNAPSHOT_DIR, f"alert_{timestamp}.jpg"
            )
            cv2.imwrite(snapshot_path, frame)

            print(f"[ALERT] {violation_reason}")
            print(f"[ALERT] Snapshot saved: {snapshot_path}")

        # ----------------------------------
        # Draw face boxes
        # ----------------------------------
        for (top, right, bottom, left), name, role in zip(
            last_face_locations, last_face_names, last_face_roles
        ):
            if role == "authorized":
                color = (0, 255, 0)
            elif role == "restricted":
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(
                frame,
                f"{name} ({role})",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # ----------------------------------
        # Draw system status
        # ----------------------------------
        if alert_active:
            cv2.putText(
                frame,
                "SECURITY ALERT",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
        else:
            cv2.putText(
                frame,
                "STATUS: ACCESS OK",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

        # ----------------------------------
        # Encode frame as JPEG
        # ----------------------------------
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "Camera server running"}

@app.get("/stream")
def stream():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
