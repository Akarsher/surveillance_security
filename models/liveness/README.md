# Liveness Model

This project now supports MiniFASNet anti-spoofing via ONNX Runtime.

Place your model in this folder with the default filename:

- `liveness_model.onnx`

If you use a different path, configure:

- `LIVENESS_MODEL_PATH`

Useful settings:

- `LIVENESS_ENABLED=true|false`
- `LIVENESS_STRICT=false|true`
- `LIVENESS_THRESHOLD=0.90`
- `LIVENESS_LIVE_CLASS_INDEX=1`
- `LIVENESS_CROP_SCALE=2.7`
- `LIVENESS_MODEL_PATH=models/liveness/liveness_model.onnx`
- `LIVENESS_MODEL_PATHS=models/liveness/liveness_model.onnx,models/liveness/liveness_model_4_0_0.onnx`
- `LIVENESS_MODEL_SCALES=2.7,4.0`

Notes:

- Official MiniFASNet test flow uses fused predictions from multiple models.
- For best spoof detection (phone replay/photo), export both official anti-spoof models and set both in `LIVENESS_MODEL_PATHS`.

Recommended validation checklist:

- Verify live camera faces return high live scores.
- Verify printed photo and mobile replay attacks return low live scores.
- Tune threshold to balance false rejects vs spoof blocking on your camera.
