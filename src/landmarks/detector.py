"""
Face landmark detector using MediaPipe Face Landmarker (Tasks API).

MediaPipe 0.10.14+ removed the legacy `mp.solutions` API.
This module uses the new Tasks API which works on all current versions.

The face_landmarker.task model (~1.4 MB) is downloaded at first use and
cached locally. It returns 478 landmarks (468 face + 10 iris).

Reference:
  Kartynnik et al., 2019 — Real-time Facial Surface Geometry from Monocular Video
  https://arxiv.org/abs/1907.06724
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "face_landmarker.task"


def _ensure_model() -> str:
    """Download model if not cached, return local path."""
    if not MODEL_PATH.exists():
        print(f"Downloading face_landmarker.task from Google Storage...")
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        print(f"Model saved to {MODEL_PATH} ({MODEL_PATH.stat().st_size // 1024} KB)")
    return str(MODEL_PATH)


# ---------------------------------------------------------------------------
# Landmark index groups (MediaPipe Face Mesh 468-point model)
# ---------------------------------------------------------------------------

LANDMARK_GROUPS = {
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "nose_tip": [1, 2, 98, 327],
    "nose_bridge": [6, 197, 195, 5],
    "upper_lip": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "lower_lip": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "jawline": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
    "midline": [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2,
                164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152],
}

# Bilateral pairs (left_idx, right_idx) across the facial midline
BILATERAL_PAIRS = [
    (33, 263),    # outer eye corners
    (133, 362),   # inner eye corners
    (159, 386),   # eye centers
    (46, 276),    # eyebrow outer
    (107, 336),   # eyebrow inner
    (234, 454),   # cheekbones
    (205, 425),   # nasolabial fold
    (61, 291),    # lip corners
    (37, 267),    # upper lip
    (172, 397),   # jaw lateral
    (136, 365),   # jaw lower
]


@dataclass
class LandmarkResult:
    """Output of FaceLandmarkDetector.detect()."""

    landmarks: np.ndarray            # (468, 2) pixel-space (x, y)
    landmarks_3d: np.ndarray         # (468, 3) normalized (x, y, z)
    image_size: tuple[int, int]      # (height, width)
    detection_confidence: float

    groups: dict[str, np.ndarray] = field(default_factory=dict)
    midline_x: float = 0.0


class FaceLandmarkDetector:
    """Face landmark detector using the MediaPipe Tasks API (mediapipe >= 0.10.14).

    Args:
        max_faces:          Maximum number of faces to detect (default 1).
        min_detection_conf: Minimum face detection confidence.
        min_presence_conf:  Minimum face presence confidence.
        min_tracking_conf:  Minimum tracking confidence.
    """

    def __init__(
        self,
        max_faces: int = 1,
        min_detection_conf: float = 0.5,
        min_presence_conf: float = 0.5,
        min_tracking_conf: float = 0.5,
        # Legacy arg kept for call-site compatibility — ignored in Tasks API
        refine_landmarks: bool = True,
    ) -> None:
        model_path = _ensure_model()

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_conf,
            min_face_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_tracking_conf,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> Optional[LandmarkResult]:
        """Detect facial landmarks in an RGB uint8 image.

        Args:
            image: (H, W, 3) uint8 numpy array in RGB colour space.

        Returns:
            LandmarkResult if a face is detected, else None.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got shape {image.shape}")

        h, w = image.shape[:2]

        # Ensure RGB uint8
        rgb = image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]   # NormalizedLandmark list (478 pts)

        # Convert normalized → pixel coordinates (use first 468 for compatibility)
        n = min(468, len(face))
        pts    = np.array([[lm.x * w, lm.y * h] for lm in face[:n]], dtype=np.float32)
        pts_3d = np.array([[lm.x, lm.y, lm.z]  for lm in face[:n]], dtype=np.float32)

        groups = {
            name: pts[np.array([i for i in indices if i < n])]
            for name, indices in LANDMARK_GROUPS.items()
        }

        midline_pts = pts[np.array([i for i in LANDMARK_GROUPS["midline"] if i < n])]
        midline_x   = float(midline_pts[:, 0].mean())

        return LandmarkResult(
            landmarks=pts,
            landmarks_3d=pts_3d,
            image_size=(h, w),
            detection_confidence=1.0,
            groups=groups,
            midline_x=midline_x,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as RGB uint8."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
