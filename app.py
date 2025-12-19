import cv2
import os
import face_recognition
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

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
# MJPEG frame generator
# -----------------------------
def generate_frames():
    process_every_n_frames = 3   # run recognition every 3 frames
    frame_count = 0

    last_face_locations = []
    last_face_names = []
    last_face_roles = []

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1

        # -----------------------------
        # Run recognition only sometimes
        # -----------------------------
        if frame_count % process_every_n_frames == 0:

            # Resize frame for faster processing
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

                # Scale back face locations to original frame size
                last_face_locations.append((
                    top * 2, right * 2, bottom * 2, left * 2
                ))
                last_face_names.append(name)
                last_face_roles.append(role)

        # -----------------------------
        # Draw last known detections
        # -----------------------------
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

            label = f"{name} ({role})"
            cv2.putText(
                frame,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # -----------------------------
        # Encode frame as JPEG
        # -----------------------------
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
