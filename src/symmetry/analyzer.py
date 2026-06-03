"""
Facial symmetry analysis.

Computes bilateral symmetry by:
  1. Reflecting all left-side landmarks across the facial midline
  2. Comparing reflected positions to the corresponding right-side landmarks
  3. Aggregating asymmetry per facial region

Symmetry score = 1 - (mean Euclidean distance of bilateral pairs / inter-ocular distance)

The inter-ocular distance (IOD) is used as the normalization baseline — it is the most
commonly used reference in clinical facial asymmetry literature (see Farkas, 1994).

Output per region:
  - raw asymmetry (pixels)
  - normalized asymmetry (fraction of IOD)
  - symmetry score [0, 1] — higher is more symmetric

Regions assessed:
  - eyes
  - eyebrows
  - cheeks
  - lips
  - jaw
  - overall

Ethical note:
  "Symmetry" here is a geometric measurement. There is no normative claim
  about what constitutes an attractive or healthy face. This tool is
  intended for research and clinical measurement purposes only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.landmarks.detector import LandmarkResult, BILATERAL_PAIRS

# Region groupings of bilateral pairs (indices into BILATERAL_PAIRS list)
REGION_PAIRS: dict[str, list[int]] = {
    "eyes":      [0, 1, 2],       # outer corner, inner corner, center
    "eyebrows":  [3, 4],
    "cheeks":    [5, 6],
    "lips":      [7, 8],
    "jaw":       [9, 10],
}


@dataclass
class SymmetryResult:
    """Output of FaceSymmetryAnalyzer.analyze()."""

    # Overall
    overall_score: float            # [0, 1]
    overall_asymmetry_px: float     # mean bilateral distance in pixels
    overall_asymmetry_iod: float    # mean bilateral distance / IOD

    # Per region
    region_scores: dict[str, float]
    region_asymmetry_px: dict[str, float]
    region_asymmetry_iod: dict[str, float]

    # Reference measurements
    inter_ocular_distance: float    # pixels
    face_width: float               # pixels (jaw-to-jaw)

    # Raw pair distances (for plotting)
    pair_distances_px: np.ndarray   # (N,) one per bilateral pair
    pair_labels: list[str]

    def summary(self) -> str:
        lines = [
            f"Overall symmetry score:  {self.overall_score:.3f}",
            f"Inter-ocular distance:   {self.inter_ocular_distance:.1f} px",
            f"Overall asymmetry:       {self.overall_asymmetry_iod*100:.1f}% of IOD",
            "",
            "Per-region scores:",
        ]
        for region, score in self.region_scores.items():
            asym_pct = self.region_asymmetry_iod.get(region, 0) * 100
            lines.append(f"  {region:12s}: {score:.3f}  (asymmetry={asym_pct:.1f}% IOD)")
        return "\n".join(lines)


class FaceSymmetryAnalyzer:
    """Compute bilateral facial symmetry from MediaPipe landmarks.

    Args:
        score_scale: Controls how asymmetry maps to [0,1] score.
            score = exp(-asymmetry_iod / score_scale)
            score_scale=0.05 means asymmetry of 5% IOD → score ≈ 0.37.
            Typical clinical asymmetry in healthy adults: 2–6% IOD.
    """

    def __init__(self, score_scale: float = 0.05) -> None:
        self.score_scale = score_scale

    def analyze(self, result: LandmarkResult) -> SymmetryResult:
        """Compute symmetry metrics from a LandmarkResult.

        Args:
            result: Output of FaceLandmarkDetector.detect().

        Returns:
            SymmetryResult with scores and per-region breakdown.
        """
        pts = result.landmarks
        midline_x = result.midline_x

        # Inter-ocular distance: outer left eye corner → outer right eye corner
        iod = float(np.linalg.norm(pts[33] - pts[263]))
        if iod < 1e-3:
            iod = 1.0  # guard against degenerate detection

        # Face width: jaw-to-jaw (approximate)
        face_width = float(np.linalg.norm(pts[234] - pts[454]))

        # Compute bilateral pair distances
        pair_distances = []
        pair_labels = [
            "outer eye corner", "inner eye corner", "eye center",
            "eyebrow outer", "eyebrow inner",
            "cheekbone", "nasolabial fold",
            "lip corner", "upper lip",
            "jaw lateral", "jaw lower",
        ]

        for left_idx, right_idx in BILATERAL_PAIRS:
            left_pt  = pts[left_idx]
            right_pt = pts[right_idx]

            # Reflect left point across midline, compare to right
            left_reflected = np.array([2 * midline_x - left_pt[0], left_pt[1]])
            dist = float(np.linalg.norm(left_reflected - right_pt))
            pair_distances.append(dist)

        pair_distances = np.array(pair_distances, dtype=np.float32)

        # Per-region aggregation
        region_scores: dict[str, float] = {}
        region_asym_px: dict[str, float] = {}
        region_asym_iod: dict[str, float] = {}

        for region, pair_indices in REGION_PAIRS.items():
            dists = pair_distances[pair_indices]
            mean_dist = float(dists.mean())
            mean_iod  = mean_dist / iod
            score     = float(np.exp(-mean_iod / self.score_scale))
            region_scores[region]      = score
            region_asym_px[region]     = mean_dist
            region_asym_iod[region]    = mean_iod

        # Overall
        overall_asym_px  = float(pair_distances.mean())
        overall_asym_iod = overall_asym_px / iod
        overall_score    = float(np.exp(-overall_asym_iod / self.score_scale))

        return SymmetryResult(
            overall_score=overall_score,
            overall_asymmetry_px=overall_asym_px,
            overall_asymmetry_iod=overall_asym_iod,
            region_scores=region_scores,
            region_asymmetry_px=region_asym_px,
            region_asymmetry_iod=region_asym_iod,
            inter_ocular_distance=iod,
            face_width=face_width,
            pair_distances_px=pair_distances,
            pair_labels=pair_labels,
        )
