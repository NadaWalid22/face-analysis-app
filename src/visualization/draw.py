"""
Visualization utilities for facial landmark and symmetry analysis.
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from src.landmarks.detector import LandmarkResult, LANDMARK_GROUPS, BILATERAL_PAIRS
from src.symmetry.analyzer import SymmetryResult


# Color palette (BGR for cv2, RGB for matplotlib)
COLORS_BGR = {
    "left_eye":      (86, 180, 233),
    "right_eye":     (86, 180, 233),
    "left_eyebrow":  (230, 159, 0),
    "right_eyebrow": (230, 159, 0),
    "nose_tip":      (0, 158, 115),
    "nose_bridge":   (0, 158, 115),
    "upper_lip":     (204, 121, 167),
    "lower_lip":     (204, 121, 167),
    "jawline":       (160, 160, 160),
    "midline":       (255, 255, 255),
}


def draw_landmarks(
    image: np.ndarray,
    result: LandmarkResult,
    radius: int = 2,
    draw_connections: bool = True,
    alpha: float = 0.7,
) -> np.ndarray:
    """Draw 468 facial landmarks on the image.

    Args:
        image:            RGB uint8 image.
        result:           LandmarkResult from FaceLandmarkDetector.
        radius:           Dot radius in pixels.
        draw_connections: Draw group contour lines.
        alpha:            Blend factor for overlay.

    Returns:
        RGB uint8 annotated image.
    """
    overlay = image.copy()
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # Draw all landmarks as small dots (gray)
    for pt in result.landmarks:
        cv2.circle(bgr, (int(pt[0]), int(pt[1])), radius, (180, 180, 180), -1)

    # Draw group landmarks with distinct colors
    for group_name, pts in result.groups.items():
        color = COLORS_BGR.get(group_name, (255, 255, 255))
        rgb_color = (color[2], color[1], color[0])  # BGR → RGB for drawing on RGB img

        for pt in pts:
            cv2.circle(bgr, (int(pt[0]), int(pt[1])), radius + 1, color, -1)

        if draw_connections and len(pts) > 1:
            for i in range(len(pts) - 1):
                p1 = (int(pts[i][0]), int(pts[i][1]))
                p2 = (int(pts[i+1][0]), int(pts[i+1][1]))
                cv2.line(bgr, p1, p2, color, 1, cv2.LINE_AA)

    # Midline
    mx = int(result.midline_x)
    h  = image.shape[0]
    cv2.line(bgr, (mx, 0), (mx, h), (200, 200, 50), 1, cv2.LINE_AA)

    result_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, result_rgb, alpha, 0)


def draw_symmetry_overlay(
    image: np.ndarray,
    lm_result: LandmarkResult,
    sym_result: SymmetryResult,
    show_pairs: bool = True,
) -> np.ndarray:
    """Draw bilateral landmark pairs color-coded by asymmetry magnitude.

    Green = low asymmetry, red = high asymmetry (relative to median).

    Args:
        image:       RGB uint8 image.
        lm_result:   LandmarkResult.
        sym_result:  SymmetryResult.
        show_pairs:  Draw connecting lines between bilateral pairs.

    Returns:
        Annotated RGB image.
    """
    bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    pts = lm_result.landmarks
    midline_x = lm_result.midline_x
    distances = sym_result.pair_distances_px

    # Normalize distances for color mapping
    d_min, d_max = distances.min(), max(distances.max(), 1.0)

    for i, (left_idx, right_idx) in enumerate(BILATERAL_PAIRS):
        left_pt  = pts[left_idx].astype(int)
        right_pt = pts[right_idx].astype(int)

        # Color: green (symmetric) → red (asymmetric)
        t = (distances[i] - d_min) / (d_max - d_min)
        color = (int(t * 255), int((1 - t) * 200), 30)  # BGR

        cv2.circle(bgr, tuple(left_pt),  4, color, -1)
        cv2.circle(bgr, tuple(right_pt), 4, color, -1)

        if show_pairs:
            cv2.line(bgr, tuple(left_pt), tuple(right_pt), color, 1, cv2.LINE_AA)

    # Midline
    cv2.line(bgr, (int(midline_x), 0), (int(midline_x), image.shape[0]),
             (80, 200, 200), 1, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_bilateral_comparison(
    image: np.ndarray,
    lm_result: LandmarkResult,
) -> Figure:
    """Create a side-by-side left/right half comparison figure.

    Flips the left half and places it next to the right half to visually
    demonstrate what a perfectly symmetric version would look like.

    Returns:
        Matplotlib Figure (3 panels: original | left-mirrored | right-mirrored)
    """
    h, w = image.shape[:2]
    mx = int(lm_result.midline_x)

    # Left half, mirrored (reflects left face to fill right side)
    left_half = image[:, :mx]
    left_mirror = np.fliplr(left_half)
    # Pad or crop to match right half width
    right_w = w - mx
    if left_mirror.shape[1] >= right_w:
        left_mirror = left_mirror[:, :right_w]
    else:
        pad = right_w - left_mirror.shape[1]
        left_mirror = np.pad(left_mirror, ((0,0),(0,pad),(0,0)))
    left_full = np.concatenate([np.fliplr(left_mirror), left_mirror], axis=1)
    left_full = left_full[:, :w]

    # Right half, mirrored
    right_half = image[:, mx:]
    right_mirror = np.fliplr(right_half)
    left_w = mx
    if right_mirror.shape[1] >= left_w:
        right_mirror = right_mirror[:, :left_w]
    else:
        pad = left_w - right_mirror.shape[1]
        right_mirror = np.pad(right_mirror, ((0,0),(0,pad),(0,0)))
    right_full = np.concatenate([right_mirror, np.fliplr(right_mirror)], axis=1)
    right_full = right_full[:, :w]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image);        axes[0].set_title('Original',        fontsize=12)
    axes[1].imshow(left_full);    axes[1].set_title('Left×2 (mirrored)', fontsize=12)
    axes[2].imshow(right_full);   axes[2].set_title('Right×2 (mirrored)', fontsize=12)
    for ax in axes:
        ax.axis('off')
    plt.suptitle('Bilateral Symmetry Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_asymmetry_radar(sym_result: SymmetryResult) -> Figure:
    """Radar / spider chart of per-region symmetry scores."""
    regions = list(sym_result.region_scores.keys())
    scores  = [sym_result.region_scores[r] for r in regions]
    n = len(regions)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    scores += scores[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, 'o-', linewidth=2, color='royalblue')
    ax.fill(angles, scores, alpha=0.25, color='royalblue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(regions, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('Per-Region Symmetry Scores', fontsize=13, pad=20)
    ax.grid(True, alpha=0.3)
    return fig


def plot_asymmetry_bar(sym_result: SymmetryResult) -> Figure:
    """Horizontal bar chart of bilateral pair asymmetries (% IOD)."""
    iod = sym_result.inter_ocular_distance
    pairs_pct = sym_result.pair_distances_px / iod * 100
    labels = sym_result.pair_labels

    colors = ['#e74c3c' if v > 5 else '#2ecc71' for v in pairs_pct]

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.5)))
    y = np.arange(len(labels))
    ax.barh(y, pairs_pct, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Asymmetry (% of inter-ocular distance)', fontsize=11)
    ax.set_title('Bilateral Pair Asymmetry', fontsize=13)
    ax.axvline(5, color='orange', linestyle='--', lw=1.5, label='5% IOD threshold')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig
