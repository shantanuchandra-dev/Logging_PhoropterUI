# Automated ROI Extraction Pipeline

A high-accuracy clinical data extraction pipeline for medical UI videos, specifically tuned for Phoropter interfaces.

## 🚀 Quick Start
```bash
python3 pipeline.py
```
Outputs are saved to the directory specified in `config.json` (default: `MatchedScreens/`).

## 🧠 Architecture: Tiered Change Detection
The pipeline uses a three-tier "Skeptical Extraction" logic to minimize redundant processing while ensuring 100% capture of clinical changes:

1.  **Frame Diff:** Checks the pixel-level difference between sampled frames.
2.  **ROI-0 Diff:** If the frame changes, it crops the main UI window (ROI-0) and compares it to the previous ROI-0. This filters out movement outside the primary interface.
3.  **Value Comparison:** Extracted clinical points (SPH, CYL, Occluder State, etc.) are compared against the last logged entry. Data is only written to CSV if a clinical value has changed.

## ⚙️ Configuration (`config.json`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `match_threshold` | float | 0.3 | Lower = more lenient UI detection. Optimized for glare/noise. |
| `frame_diff_threshold` | float | 0.0002 | Sensitivity for detecting frame-level motion (0.02%). |
| `roi0_diff_threshold` | float | 0.0005 | Sensitivity for detecting UI-level changes (0.05%). |
| `sampling_interval_seconds`| float | 2.0 | Interval between sampled frames. |
| `reference_image` | str | "topcon_ui_001.png" | Template used to "lock on" to the UI. |
| `save_debug_images` | bool | true | Saves annotated ROI crops for verification. |
| `max_consecutive_failures` | int | 10 | Limit for failed extraction attempts before warning. |

## 📊 Extracted Regions
- **ROI-1 (OCR):** Precise SPH, CYL, AXIS, and ADD values for both eyes.
- **ROI-2:** Pupil Distance (PD) values.
- **ROI-3/4:** Phoropter Occluder states (BINO, Left_Occluded, Axis_Flip, etc.).
- **ROI-5:** Active Chart Tab detection.
- **ROI-7:** Clinical Chart display identification.

## 🛠️ Requirements
- **Python:** 3.8+
- **System:** `Tesseract OCR` (must be installed on the host OS)
- **PIPs:** `opencv-python`, `pytesseract`, `easyocr`, `numpy`, `torch`

## 🖥️ GPU Support
Identified GPUs will be utilized for JCC pattern classification. CPU fallback is automatic and fully supported.
