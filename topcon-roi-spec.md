# UI Image Analysis – ROI Specification and Extraction Pipeline

## Coordinate system and ROI‑0

- Crop ROI‑0 to **only** the medical application window, excluding desktop, taskbar, TeamViewer, and Zoho Assist overlays.
- All other ROI coordinates are \((x, y, width, height)\) **integers**, relative to the top‑left of ROI‑0, with a 3–5 px padding after edge-based refinement.
- Use edge detection (e.g., Canny) + contours to snap bounding boxes to visual boundaries, then expand by padding before OCR/feature extraction.

## ROI definitions and expected outputs

### ROI‑1 – S/C/A/ADD table

- Region: central white/black (color may vary) table only, excluding gray background.
- Task: OCR structured values:
  - Columns: Right eye, Left eye.
  - Rows: S, C, A, ADD.
- Method:
  - Edge detection + contour grouping to isolate the rectangular table.
  - Optionally template-align to a reference table layout, then define sub-ROIs for each cell and use OCR (e.g., Tesseract/EasyOCR) for text.
- Output per frame example:
  ```json
  {
    "R": {"S": "0.00", "C": "-0.00", "A": "180", "ADD": "0.00"},
    "L": {"S": "0.00", "C": "-0.00", "A": "180", "ADD": "0.00"},
    "confidences": {...}
  }
  ```

### ROI‑2 – PD label and value

- Region: the small rectangle containing **“PD”** and the numeric value below (e.g., "64.0"), no circles.
- Task: OCR PD numeric value; ignore surrounding UI.
- Method:
  - Locate via template matching on the "PD" label or by fixed relative position inside ROI‑0 after initial alignment.
  - Use OCR on the value cell; apply numeric post‑processing (allow only digits and dot).
- Output example:
  ```json
  {"pd_value": 64.0, "confidence": 0.95}
  ```

### ROI‑3 and ROI‑4 – left_occluder/right_occluder circles

- Region: primarily the **circular** button graphics, as tightly as edge detection allows.
- Task: classify **filled/unfilled** state (no text extraction).
- Method:
  - Detect circles via contours + `minEnclosingCircle` or template matching; then crop each circle.
  - Train a lightweight binary classifier (or use heuristics based on mean intensity/color) to label filled vs unfilled.
- Output example:
  ```json
  {
    "left_occluder": {"state": "filled", "confidence": 0.93},
    "right_occluder": {"state": "unfilled", "confidence": 0.90}
  }
  ```

### ROI‑5 – Chart tabs ("Chart 1–5")

- Region: the horizontal strip containing the tab buttons "Chart1" … "Chart5".
- Task: determine which chart tab is selected (e.g., highlighted or different background).
- Method:
  - OCR each tab label and/or detect style differences.
  - Or template-match on the selected tab style (color/border) and map its x-position to a tab index.
- Output example:
  ```json
  {"selected_chart_tab": "Chart1", "index": 0}
  ```

### ROI‑6 – Chart options grid below tabs (A + B)

- Region: only the grid of small chart thumbnails under the tabs, excluding the lower status bar.
- Task:
  - A) Return bounding boxes for **all** thumbnails plus a dedicated ROI for the **selected** thumbnail.
  - B) Also expose a single overall ROI‑6 box and a `selected_index`.
- Method:
  - Use edge/contour detection to segment rectangular thumbnails; enforce grid regularity (rows/columns).
  - Determine the selected one by distinct border/colors, or via similarity to a known "selected" template.
- Output example:
  ```json
  {
    "roi6_bbox": [x, y, w, h],
    "thumbnails": [
      {"bbox": [x1, y1, w1, h1]},
      {"bbox": [x2, y2, w2, h2]},
      ...
    ],
    "selected_index": 3,
    "selected_bbox": [xs, ys, ws, hs]
  }
  ```

### ROI‑7 – Big chart on the right

- Region: the big chart pane containing the character (e.g., "M").
- Task: detect the displayed character/symbol.
- Method options:
  - OCR if characters are high-contrast and font-like.
  - Or train a classifier on chart symbols (e.g., E, M, rings, etc.) using cropped ROI‑7 images.
- Output example:
  ```json
  {"chart_symbol": "M", "confidence": 0.97}
  ```

### ROI‑8 – SPH tool (R/BINO/L + SPH)

- Region: full yellow framed container including R, BINO, L buttons and the "(–) SPH (+)" panel.
- Task:
  - Detect which of R/BINO/L is active.
  - Determine SPH sign/state (and numeric value if present).
  - Handle case where this modal is not visible.
- Method:
  - First, detect presence of ROI‑8 via template matching or color/shape cues; if not found, mark `visible=false`.
  - When visible, segment button areas and classify each as selected/unselected based on color/intensity.
  - OCR or parse any SPH text if available.
- Output example:
  ```json
  {
    "visible": true,
    "mode": "BINO",
    "sph": "+1.00",
    "confidence": 0.92
  }
  ```

## Dynamic / real‑time behavior and screen filtering

- Capture the window region for ROI‑0 continuously (e.g., 2–5 fps).
- Before running extraction, verify that the medical UI is present (template match or key feature detection). If not, treat the frame as **ignore**.
- For each valid frame:
  - Align ROI‑0 to a reference template (homography or feature-based alignment) to stabilize layout.
  - Apply the ROI finders above (table, PD, occluders, chart tabs, grid, big chart, SPH tool).
- Emit a structured JSON state per frame containing ROI coordinates, extracted values, and per-field confidences. This can drive monitoring, automation, or further analytics.
