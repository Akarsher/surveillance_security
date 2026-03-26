# Liveness Model

Place your ONNX anti-spoof model in this folder using the default name:

- liveness_model.onnx

If you use another filename/path, set `LIVENESS_MODEL_PATH` in your environment.

Recommended validation checklist:
- Verify live camera faces return high live scores.
- Verify printed photo and mobile replay attacks return low live scores.
- Tune `LIVENESS_THRESHOLD` to balance false rejects vs spoof blocking.
