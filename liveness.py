import os
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np
import onnxruntime as ort


class MiniFASNetLiveness:
    """ONNX runtime wrapper for MiniFASNet anti-spoofing inference."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.8,
        live_class_index: int = 1,
        strict: bool = True,
        enabled: bool = True,
        crop_scale: float = 2.7,
        model_scales: Optional[Iterable[float]] = None,
        providers: Optional[Iterable[str]] = None,
    ) -> None:
        self.model_paths = self._normalize_model_paths(model_path)
        self.threshold = float(threshold)
        self.live_class_index = int(live_class_index)
        self.strict = bool(strict)
        self.enabled = bool(enabled)
        self.crop_scale = float(crop_scale)
        self.model_scales = self._normalize_model_scales(model_scales)

        self.ready = False
        self.error_message: Optional[str] = None
        self.sessions = []

        if not self.enabled:
            return

        try:
            providers_list = list(providers) if providers else None

            for idx, path in enumerate(self.model_paths):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found: {path}")

                session = ort.InferenceSession(path, providers=providers_list)
                input_meta = session.get_inputs()[0]

                input_height = 80
                input_width = 80
                if len(input_meta.shape) >= 4:
                    input_height = self._safe_int(input_meta.shape[-2], 80)
                    input_width = self._safe_int(input_meta.shape[-1], 80)

                model_scale = self.model_scales[idx] if idx < len(self.model_scales) else self.crop_scale
                self.sessions.append(
                    {
                        "path": path,
                        "session": session,
                        "input_name": input_meta.name,
                        "input_height": input_height,
                        "input_width": input_width,
                        "scale": float(model_scale),
                    }
                )

            self.ready = len(self.sessions) > 0
        except Exception as exc:  # pragma: no cover - runtime setup guard
            self.error_message = f"Failed to initialize liveness model: {exc}"
            self.ready = False

    @staticmethod
    def _normalize_model_paths(raw: str) -> list:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if not parts:
            return []
        return [os.path.abspath(p) for p in parts]

    @staticmethod
    def _normalize_model_scales(raw: Optional[Iterable[float]]) -> list:
        if raw is None:
            return []
        if isinstance(raw, str):
            vals = [v.strip() for v in raw.split(",") if v.strip()]
        else:
            vals = list(raw)

        out = []
        for v in vals:
            try:
                out.append(float(v))
            except Exception:
                continue
        return out

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        x = logits.astype(np.float32)
        x = x - np.max(x)
        ex = np.exp(x)
        denom = np.sum(ex)
        if denom <= 0:
            return np.zeros_like(x)
        return ex / denom

    @staticmethod
    def _detect_frame_quality(image: np.ndarray, bbox: Any) -> Dict[str, Any]:
        """Detect frame quality issues: blur, low contrast, corruption, flat surfaces (phone screens)."""
        if bbox is None or len(bbox) < 4:
            return {"quality_ok": False, "issues": ["invalid_bbox"]}

        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return {"quality_ok": False, "issues": ["invalid_crop"]}

        face_roi = image[y1:y2, x1:x2]
        if face_roi.size == 0:
            return {"quality_ok": False, "issues": ["empty_crop"]}

        issues = []

        # Check for blur using Laplacian variance
        laplacian_var = None
        if face_roi.shape[0] > 10 and face_roi.shape[1] > 10:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:  # Low variance = likely blurred
                issues.append("blur_detected")

        # Check for low contrast
        if face_roi.size > 0:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            contrast = gray.std()
            if contrast < 15:  # Very low std = likely washed out or stuck
                issues.append("low_contrast")

        # Check for uniform color (stuck frame or flat surface like phone screen)
        if face_roi.size > 0 and len(face_roi.shape) == 3:
            # Reshape to get unique colors more efficiently
            pixels = face_roi.reshape(-1, 3)
            unique_colors = len(np.unique(pixels, axis=0))
            total_pixels = face_roi.shape[0] * face_roi.shape[1]
            
            # Phone screens and printed images have very low color diversity
            if unique_colors < total_pixels * 0.08:  # <8% color variety = likely non-face
                issues.append("flat_surface_or_uniform")
            
            # Additional check: phone screens often have limited color ranges
            # Check for artificial/printed colors (high saturation patterns)
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            # Phone screens often have very high saturation spikes (printed colors)
            high_sat_pixels = np.sum(saturation > 240)
            if high_sat_pixels > total_pixels * 0.15:  # >15% pixels with extreme saturation
                issues.append("artificial_colors_detected")

        quality_ok = len(issues) == 0
        return {"quality_ok": quality_ok, "issues": issues, "blur_var": laplacian_var}

    def _crop_face(self, image: np.ndarray, bbox: Any, scale: float) -> Optional[np.ndarray]:
        if bbox is None or len(bbox) < 4:
            return None

        h, w = image.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        nw = bw * scale
        nh = bh * scale

        nx1 = max(0, int(round(cx - nw / 2.0)))
        ny1 = max(0, int(round(cy - nh / 2.0)))
        nx2 = min(w, int(round(cx + nw / 2.0)))
        ny2 = min(h, int(round(cy + nh / 2.0)))

        if nx2 <= nx1 or ny2 <= ny1:
            return None

        return image[ny1:ny2, nx1:nx2]

    def _preprocess(self, face_crop_bgr: np.ndarray, input_width: int, input_height: int) -> np.ndarray:
        resized = cv2.resize(face_crop_bgr, (input_width, input_height))
        # MiniFASNet reference inference consumes OpenCV BGR pixels as float32
        # without channel swap or [0,1] normalization.
        tensor = resized.astype(np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    def predict(self, image: np.ndarray, bbox: Any) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "ok": True,
                "is_live": True,
                "score": 1.0,
                "threshold": self.threshold,
                "reason": "disabled",
                "frame_quality": {"quality_ok": True, "issues": []},
            }

        if not self.ready or not self.sessions:
            return {
                "ok": False,
                "is_live": not self.strict,
                "score": 0.0,
                "threshold": self.threshold,
                "reason": self.error_message or "model not ready",
                "frame_quality": {"quality_ok": False, "issues": ["model_not_ready"]},
            }

        try:
            # Check frame quality first
            frame_quality = self._detect_frame_quality(image, bbox)

            fused_probs = None
            per_model = []

            for entry in self.sessions:
                face_crop = self._crop_face(image, bbox, entry["scale"])
                if face_crop is None or face_crop.size == 0:
                    continue

                input_tensor = self._preprocess(face_crop, entry["input_width"], entry["input_height"])
                raw_output = entry["session"].run(None, {entry["input_name"]: input_tensor})
                logits = np.asarray(raw_output[0]).squeeze()
                if logits.ndim > 1:
                    logits = logits[0]
                logits = logits.astype(np.float32).reshape(-1)

                probs = self._softmax(logits)
                if probs.size == 0:
                    continue

                if fused_probs is None:
                    fused_probs = probs.copy()
                else:
                    if fused_probs.shape != probs.shape:
                        return {
                            "ok": False,
                            "is_live": not self.strict,
                            "score": 0.0,
                            "threshold": self.threshold,
                            "reason": "model output class mismatch",
                            "frame_quality": frame_quality,
                        }
                    fused_probs += probs

                per_model.append(
                    {
                        "path": entry["path"],
                        "scale": entry["scale"],
                        "probs": probs.tolist(),
                    }
                )

            if fused_probs is None or len(per_model) == 0:
                return {
                    "ok": False,
                    "is_live": not self.strict,
                    "score": 0.0,
                    "threshold": self.threshold,
                    "reason": "invalid face crop or empty model output",
                    "frame_quality": frame_quality,
                }

            label = int(np.argmax(fused_probs))
            live_idx = int(np.clip(self.live_class_index, 0, fused_probs.size - 1))
            # Match official code style: summed confidence normalized by model count.
            live_score = float(fused_probs[live_idx] / len(per_model))
            is_live = (label == live_idx) and (live_score >= self.threshold)

            return {
                "ok": True,
                "is_live": is_live,
                "score": live_score,
                "threshold": self.threshold,
                "live_class_index": live_idx,
                "predicted_label": label,
                "class_count": int(fused_probs.size),
                "model_count": len(per_model),
                "fused_probs": (fused_probs / len(per_model)).tolist(),
                "per_model": per_model,
                "reason": "ok",
                "frame_quality": frame_quality,
            }
        except Exception as exc:  # pragma: no cover - runtime inference guard
            return {
                "ok": False,
                "is_live": not self.strict,
                "score": 0.0,
                "threshold": self.threshold,
                "reason": f"inference failed: {exc}",
                "frame_quality": {"quality_ok": False, "issues": ["inference_error"]},
            }
