"""
GradioFaceAnalysis — facial symmetry analysis demo.

Deployed at: https://huggingface.co/spaces/naddaa/face-analysis-app

Run locally:
    python demo/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"
EXAMPLES = sorted([str(p) for p in SAMPLE_DIR.glob("example_*.jpg")])

from src.landmarks.detector import FaceLandmarkDetector, load_image
from src.symmetry.analyzer import FaceSymmetryAnalyzer
from src.visualization.draw import (
    draw_landmarks,
    draw_symmetry_overlay,
    draw_bilateral_comparison,
    plot_asymmetry_radar,
    plot_asymmetry_bar,
)

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

detector = FaceLandmarkDetector(max_faces=1, refine_landmarks=True)
analyzer = FaceSymmetryAnalyzer(score_scale=0.05)


def analyze(image_np: np.ndarray) -> tuple:
    """Full pipeline: image → landmark detection → symmetry analysis → visuals.

    Args:
        image_np: RGB uint8 image from Gradio (H, W, 3).

    Returns:
        Tuple of outputs: (landmark_img, overlay_img, bilateral_fig,
                           radar_fig, bar_fig, text_report)
    """
    if image_np is None:
        raise gr.Error("Please upload an image.")

    # Ensure uint8
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

    lm_result = detector.detect(image_np)
    if lm_result is None:
        raise gr.Error(
            "No face detected. Please use a clear, front-facing portrait "
            "with good lighting and the full face visible."
        )

    sym_result = analyzer.analyze(lm_result)

    # Visualizations
    landmark_img    = draw_landmarks(image_np, lm_result)
    overlay_img     = draw_symmetry_overlay(image_np, lm_result, sym_result)
    bilateral_fig   = draw_bilateral_comparison(image_np, lm_result)
    radar_fig       = plot_asymmetry_radar(sym_result)
    bar_fig         = plot_asymmetry_bar(sym_result)

    # Text report
    report = build_report(sym_result)

    return landmark_img, overlay_img, bilateral_fig, radar_fig, bar_fig, report


def build_report(sym: "SymmetryResult") -> str:
    score_pct = sym.overall_score * 100
    iod_asym  = sym.overall_asymmetry_iod * 100

    lines = [
        "## 📊 Symmetry Analysis Report",
        "",
        f"**Overall symmetry score:** {score_pct:.1f} / 100",
        f"**Overall asymmetry:** {iod_asym:.1f}% of inter-ocular distance",
        f"**Inter-ocular distance:** {sym.inter_ocular_distance:.0f} px",
        "",
        "### Per-Region Breakdown",
        "",
        "| Region | Symmetry Score | Asymmetry (% IOD) |",
        "|--------|---------------|-------------------|",
    ]
    for region, score in sym.region_scores.items():
        asym_pct = sym.region_asymmetry_iod[region] * 100
        bar = "🟢" if score > 0.7 else ("🟡" if score > 0.4 else "🔴")
        lines.append(f"| {region.capitalize()} | {bar} {score:.3f} | {asym_pct:.1f}% |")

    lines += [
        "",
        "### Interpretation",
        "",
        "- **Score > 0.7** — low asymmetry, within typical range for most people",
        "- **Score 0.4–0.7** — moderate asymmetry, common and clinically unremarkable",
        "- **Score < 0.4** — notable asymmetry; may reflect pose angle, lighting, or anatomy",
        "",
        "---",
        "",
        "⚠️ **Limitations & Ethics**",
        "",
        "This tool measures *geometric* bilateral symmetry only. It does not:",
        "- Make any claims about attractiveness, health, or normality",
        "- Account for natural head pose variation or camera angle",
        "- Replace clinical facial assessment (Bell's palsy, stroke, congenital conditions)",
        "- Work reliably on non-frontal images, partial occlusion, or extreme lighting",
        "",
        "Perfect symmetry (score=1.0) is not a goal — human faces are naturally asymmetric,",
        "and slight asymmetry is associated with individual identity and expression.",
        "",
        "Do not use this tool to make medical diagnoses.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(), title="Face Symmetry Analysis") as demo:
    gr.Markdown(
        """
        # 🔍 Face Symmetry Analyzer
        Upload a **front-facing portrait** to analyze facial bilateral symmetry using MediaPipe Face Mesh.

        > **Privacy**: images are processed locally and never stored.
        """
    )

    with gr.Row():
        input_image = gr.Image(label="Upload portrait", type="numpy", height=400)

    if EXAMPLES:
        gr.Examples(
            examples=EXAMPLES,
            inputs=input_image,
            label="Try an example (AI-generated faces — no real people)",
        )

    with gr.Row():
        analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

    gr.Markdown("---")

    with gr.Tabs():
        with gr.Tab("Landmarks"):
            landmark_out = gr.Image(label="468 facial landmarks", height=400)
        with gr.Tab("Symmetry Overlay"):
            overlay_out = gr.Image(label="Bilateral pairs (green=symmetric, red=asymmetric)", height=400)
        with gr.Tab("Bilateral Comparison"):
            bilateral_out = gr.Plot(label="Left×2 vs Original vs Right×2")
        with gr.Tab("Radar Chart"):
            radar_out = gr.Plot(label="Per-region symmetry scores")
        with gr.Tab("Asymmetry Bar"):
            bar_out = gr.Plot(label="Bilateral pair asymmetry (% IOD)")
        with gr.Tab("Report"):
            report_out = gr.Markdown()

    analyze_btn.click(
        fn=analyze,
        inputs=[input_image],
        outputs=[landmark_out, overlay_out, bilateral_out, radar_out, bar_out, report_out],
    )

    gr.Markdown(
        """
        ---
        **How it works:**
        1. **MediaPipe Face Mesh** detects 468 3D facial landmarks
        2. Each left-side landmark is reflected across the facial midline
        3. The reflected position is compared to the corresponding right-side landmark
        4. Asymmetry = mean Euclidean distance, normalized by inter-ocular distance (IOD)
        5. Score = exp(-asymmetry / 0.05) — exponential decay from perfect symmetry

        **References:**
        - Kartynnik et al. (2019) — [Real-time Facial Surface Geometry from Monocular Video](https://arxiv.org/abs/1907.06724)
        - Farkas, L.G. (1994) — *Anthropometry of the Head and Face*, 2nd ed.
        """
    )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
