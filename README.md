# Automated ROI Extraction Pipeline

A high-accuracy clinical data extraction pipeline for medical UI videos, specifically tuned for Phoropter interfaces. This system is optimized for high-performance parallel processing on Apple M4 hardware.

---

## 🛠️ End-to-End Workflow

### 1. Download Videos
Use the downloader script to fetch videos from the master dataset.

```bash
# Download a specific range of videos (e.g., from SNo 301 to 855)
python3 download_videos.py --from 301 --till 855
```
*   **Input**: `Sample/AI Optom Co-Pilot - Dataset + Trackr - Consolidated 800.csv`
*   **Output**: Videos are saved to `Sample/videos-2/` (automatically checks for duplicates in `Sample/videos/`).

### 2. Run ROI Extraction
Process the downloaded videos to extract clinical data and individual interface components.

```bash
# Process all videos in the configured directory (Parallel)
python3 pipeline.py

# Process a single video file
python3 pipeline.py <path_to_video.mp4>
```
*   **Hardware**: Utilizes all logical CPU cores (10 cores on M4) and GPU acceleration via Apple Metal (MPS).
*   **Outputs**: 
    *   `MatchedScreens/`: CSV results and coordinate metadata.
    *   `ROI_*/`: Visual crops of specific components (OCR fields, occluders, tabs, charts).

### 3. Post-Process & Clean Data
Analyze the raw extraction results to fix OCR errors, fill missing frames, and merge findings into the master dataset.

```bash
# Run the clinical analysis and master merge
python3 analyze_outputs.py
```
*   **Intelligent Correction**: Automatically fixes OCR digit misreads (e.g., `9` -> `0`) and fills 1-2 frame gaps using temporal interpolation.
*   **JCC Sequencing**: Validates clinical workflows (ensuring Flip 1 precedes Flip 2).
*   **Dataset Integration**: Appends `Exam_Status` (COMPLETE/PARTIAL/POOR) and JCC fulfillment metrics directly to the `Consolidated 800.csv` master file.
*   **Clean Output**: Finalized CSVs are saved to `Analyzed_CSVs/`.

---

## ⚙️ Configuration (`config.json`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `match_threshold` | float | 0.3 | Lower = more lenient UI detection. Optimized for glare/noise. |
| `frame_diff_threshold` | float | 0.0002 | Sensitivity for detecting frame-level motion (0.02%). |
| `roi0_diff_threshold` | float | 0.0005 | Sensitivity for detecting UI-level changes (0.05%). |
| `sampling_interval_seconds`| float | 2.0 | Interval between sampled frames. |
| `reference_image` | str | "topcon_ui_001.png" | Template used to "lock on" to the UI. |
| `save_debug_images` | bool | true | Saves annotated ROI crops for verification. |

---

## 📊 Extracted Regions
- **ROI-1 (OCR):** Precise SPH, CYL, AXIS, and ADD values for both eyes.
- **ROI-2:** Pupil Distance (PD) values.
- **ROI-3/4:** Phoropter Occluder states (BINO, Left_Occluded, Axis_Flip, etc.).
- **ROI-5:** Active Chart Tab detection.
- **ROI-7:** Clinical Chart identification.

---

## 🚦 Order of Execution

1. **Create files in MatchedScreens**
   - Run: `python3 pipeline.py`
   - Output: CSVs and metadata in `MatchedScreens/`

2. **Create files in Analyzed_CSVs**
   - Run: `python3 analyze_outputs.py`
   - Output: Cleaned and deduped CSVs in `Analyzed_CSVs/`

3. **Create files in Valid CSVs**
   - Run: `python3 filter_valid_csvs.py`
   - Output: Filtered/validated CSVs in the appropriate folder (e.g., `Valid_CSVs/`)

4. **Create files in Curated Conversations**
   - Run: `python3 curate_conversations.py`
   - Output: Curated conversation files in `Curated_Conversations/`

---

## 🏃‍♂️ Quick Start
To process a new dataset, run the following scripts in order:

```bash
python3 pipeline.py
python3 analyze_outputs.py
python3 filter_valid_csvs.py
python3 curate_conversations.py
```

Each step produces files for the next stage. See above for output folders.

---

## 💻 Requirements
- **Python**: 3.13.9+
- **System**: `Tesseract OCR` (must be installed on the host OS)
- **Environment**: Use the provided `.venv` or install from `requirements.txt`.
