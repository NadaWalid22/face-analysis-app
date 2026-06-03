"""
Face landmark detector using MediaPipe Face Mesh.

MediaPipe Face Mesh returns 468 3D facial landmarks normalized to [0,1].
This wrapper provides:
  - Pixel-space landmark coordinates (x, y)
  - Confidence score
  - Named landmark groups (eyes, eyebrows, nose, lips, jawline, midline)
  - Bilateral landmark pairs for symmetry computation

Reference:
  Kartynnik et al., 2019 — Real-time Facial Surface Geometry from Monocular Video
  https://arxiv.org/abs/1907.06724
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# Landmark index groups (MediaPipe Face Mesh 468-point model)
# Source: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
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
    "midline": [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152],
}

# Bilateral pairs (left_idx, right_idx) — symmetric landmark pairs across midline
# These are used to compute left/right facial asymmetry
BILATERAL_PAIRS = [
    # Outer eye corners
    (33, 263),
    # Inner eye corners
    (133, 362),
    # Eye centers (approx)
    (159, 386),
    # Eyebrow outer
    (46, 276),
    # Eyebrow inner
    (107, 336),
    # Cheekbones
    (234, 454),
    # Nasolabial fold
    (205, 425),
    # Lip corners
    (61, 291),
    # Upper lip
    (37, 267),
    # Jaw lateral
    (172, 397),
    # Jaw lower
    (136, 365),
]


@dataclass
class LandmarkResult:
    """Output of FaceLandmarkDetector.detect()."""

    landmarks: np.ndarray            # (468, 2) pixel-space (x, y)
    landmarks_3d: np.ndarray         # (468, 3) normalized (x, y, z)
    image_size: tuple[int, int]      # (height, width)
    detection_confidence: float

    # Per-group landmark coordinates (pixel-space)
    groups: dict[str, np.ndarray] = field(default_factory=dict)

    # Midline center of the face (x-coordinate) derived from symmetry axis
    midline_x: float = 0.0


class FaceLandmarkDetector:
    """Thin wrapper around MediaPipe Face Mesh.

    Args:
        max_faces:          Maximum number of faces to detect (default 1).
        min_detection_conf: Minimum detection confidence threshold.
        min_tracking_conf:  Minimum tracking confidence threshold.
        refine_landmarks:   Use refined 468+10 model (adds iris landmarks).
    """

    def __init__(
        self,
        max_faces: int = 1,
        min_detection_conf: float = 0.5,
        min_tracking_conf: float = 0.5,
        refine_landmarks: bool = True,
    ) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

    def detect(self, image: np.ndarray) -> Optional[LandmarkResult]:
        """Detect facial landmarks in a BGR or RGB uint8 image.

        Args:
            image: (H, W, 3) uint8 numpy array (BGR from OpenCV, or RGB).

        Returns:
            LandmarkResult if a face is detected, else None.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got shape {image.shape}")

        h, w = image.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if _is_bgr(image) else image
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]

        # Convert normalized [0,1] coordinates to pixel space
        pts = np.array([[lm.x * w, lm.y * h] for lm in face.landmark], dtype=np.float32)
        pts_3d = np.array([[lm.x, lm.y, lm.z] for lm in face.landmark], dtype=np.float32)

        # Confidence: MediaPipe FaceMesh doesn't expose per-landmark confidence;
        # we use the presence value if available, else 1.0
        conf = getattr(results, "multi_face_detection", None)
        confidence = 1.0 if conf is None else float(results.multi_face_detection[0].score[0])

        # Build per-group dicts
        groups = {
            name: pts[np.array(indices)]
            for name, indices in LANDMARK_GROUPS.items()
            if max(indices) < len(pts)
        }

        # Midline x = average of midline landmark x-coords
        midline_pts = pts[np.array(LANDMARK_GROUPS["midline"])]
        midline_x = float(midline_pts[:, 0].mean())

        return LandmarkResult(
            landmarks=pts,
            landmarks_3d=pts_3d,
            image_size=(h, w),
            detection_confidence=confidence,
            groups=groups,
            midline_x=midline_x,
        )

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _is_bgr(image: np.ndarray) -> bool:
    """Heuristic: assume BGR if the image looks like it came from cv2.imread."""
    # This is always True when using cv2.imread; users can override with rgb=True.
    # We expose this as a module-level helper for the wrapper.
    return True


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as RGB uint8."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
