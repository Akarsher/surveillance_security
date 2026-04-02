# 🔐 Surveillance Security System

The **Surveillance Security System** is a real-time monitoring solution designed to enhance security in restricted areas. It leverages advanced **face recognition technology** to identify individuals and enforce access control policies.

The system detects unauthorized access, generates instant alerts, logs all events, and provides an intuitive dashboard for administrators and employees.

---

## 🚀 Features

### 🔹 Core Functionalities

- **Real-Time Face Detection**
  - Detects all faces from live camera feed using computer vision.

- **Face Recognition & Verification**
  - Matches detected faces with stored embeddings of authorized/restricted users.

- **Liveness Detection (MiniFASNet, ONNX)**
  - Blocks spoof attempts (photo/screen replay) during face login and face attendance.

- **Access Control System**
  - Alerts when:
    - Restricted personnel enter unauthorized zones
    - Unknown individuals appear
    - Suspicious group entries occur

- **Automatic Alert Generation**
  - Buzzer / sound alert
  - Dashboard notifications
  - Snapshot capture for evidence

- **Logging & Reporting**
  - Time-stamped logs for all activities
  - Export logs as CSV

- **User Dashboards**
  - **Admin Dashboard**
    - Manage users (add/edit/delete)
    - Monitor live feed
    - View alerts and reports
  - **Employee Dashboard**
    - View profile and logs
    - Update personal details

- **SMS Notifications**
  - Integrated with Twilio API
  - Sends alerts for critical security events

- **Scalable Architecture**
  - Supports multiple cameras
  - Configurable access policies

---

## 🛠️ Tech Stack

- **Backend:** FastAPI (Python)
- **Computer Vision:** OpenCV, Face Recognition
- **Database:** MongoDB / SQL (based on implementation)
- **Notifications:** Twilio API
- **Frontend:** HTML, CSS, JavaScript
- **Environment Management:** Python-dotenv

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-repo/surveillance-security-system.git
cd surveillance-security-system
pip install -r requirements.txt
uvicorn app:app --reload

### Liveness Model Setup

Place your MiniFASNet ONNX model at:

`models/liveness/liveness_model.onnx`

Optional environment variables:

- `LIVENESS_ENABLED=true`
- `LIVENESS_STRICT=false`
- `LIVENESS_MODEL_PATH=models/liveness/liveness_model.onnx`
- `LIVENESS_THRESHOLD=0.80`
- `LIVENESS_LIVE_CLASS_INDEX=1`
- `LIVENESS_CROP_SCALE=2.7`