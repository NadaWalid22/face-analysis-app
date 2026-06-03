---
title: Face Symmetry Analyzer
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
app_file: demo/app.py
pinned: false
license: mit
short_description: Bilateral facial symmetry analysis using MediaPipe Face Mesh
---

# face-analysis-app

> Facial landmark detection and bilateral symmetry analysis using MediaPipe Face Mesh (468 3D landmarks). Measures per-region geometric asymmetry normalized to inter-ocular distance.

---

## Demo

**Live demo:** [face-analysis-app on Hugging Face Spaces](https://huggingface.co/spaces/naddaa/face-analysis-app)

**Run locally:**
```bash
pip install -r requirements.txt
python demo/app.py
```

Upload a front-facing portrait and get:
- 468-point facial landmark overlay
- Bilateral pair asymmetry map (green = symmetric, red = asymmetric)
- Left/right mirror comparison
- Per-region symmetry radar chart
- Asymmetry bar chart (% of inter-ocular distance)

---

## How It Works

```
Input: front-facing portrait image
         ↓
MediaPipe Face Mesh (468 3D landmarks)
         ↓
Midline estimation (average of medial landmarks)
         ↓
Bilateral reflection:
  left_reflected = (2 × midline_x − left_x, left_y)
  asymmetry = ||left_reflected − right||₂
         ↓
Normalization by inter-ocular distance (IOD)
         ↓
Per-region aggregation (eyes, brows, cheeks, lips, jaw)
         ↓
Score = exp(−asymmetry_IOD / 0.05)
```

The inter-ocular distance (IOD) normalization follows clinical convention (Farkas, 1994) — it makes the metric scale-independent across different image resolutions and face sizes.

---

## Symmetry Score Interpretation

| Score | Interpretation |
|-------|----------------|
| > 0.7 | Low asymmetry — within typical range |
| 0.4–0.7 | Moderate asymmetry — common, clinically unremarkable |
| < 0.4 | Notable asymmetry — may reflect pose, lighting, or anatomy |

**Note:** Typical clinical asymmetry in healthy adults is 2–6% IOD. Perfect symmetry (score = 1.0) is neither expected nor a marker of attractiveness.

---

## Ethical Limitations

This tool measures **geometric bilateral symmetry only**. It:

- Makes **no claims** about attractiveness, health, normality, or identity
- Does **not** work reliably on non-frontal images or partial occlusion
- Should **not** replace clinical assessment of facial palsy, stroke, or congenital conditions
- Cannot distinguish between genuine anatomical asymmetry and measurement artifacts from head pose

Human faces are naturally asymmetric. Slight asymmetry is normal and even contributes to individual facial identity. The tool is designed for research and education — not for self-assessment or clinical diagnosis.

---

## Architecture

```
src/
├── landmarks/
│   └── detector.py          # MediaPipe Face Mesh wrapper, 468 landmarks
├── symmetry/
│   └── analyzer.py          # Bilateral reflection + per-region asymmetry
└── visualization/
    └── draw.py              # cv2 + matplotlib: overlays, radar, bar chart

demo/
└── app.py                   # Gradio interface (HF Spaces)
```

---

## References

- Kartynnik, Y. et al. (2019). Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs. *arXiv:1907.06724*.
- Farkas, L.G. (1994). *Anthropometry of the Head and Face*, 2nd ed. Raven Press.
- Luijkx, M. et al. (2019). Quantitative facial asymmetry: Development of a 3D photogrammetric software tool. *Journal of Cranio-Maxillo-Facial Surgery*.
