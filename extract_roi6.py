#!/usr/bin/env python3
"""
Extract ROI-6 Chart Options Grid
1. Finds the Main Grid Anchor using shape detection (no template needed).
2. Uses Edge Projections to detect the internal grid cells (thumbnails).
3. Automatically adapts to 5x4 or 5x3 grids.
"""

import cv2
import numpy as np
import json
from pathlib import Path

# --- Shared Logic from ROI-5 (The Anchor) ---
def find_left_grid_anchor(img):
    """
    Finds the chart grid in the bottom-left area.
    """
    h_img, w_img = img.shape[:2]
    
    # 1. STRICT ROI: Bottom area below tabs
    search_y_start = int(h_img * 0.35) 
    search_x_end = w_img  # Full width
    
    crop = img[search_y_start:h_img, 0:search_x_end]
    
    # 2. Morphological Block Finding
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=4)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_grid = None
    max_area = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Must be big enough to be the chart grid
        if area > 10000:
            if area > max_area:
                max_area = area
                real_y = search_y_start + y
                # Add padding to include borders
                best_grid = (x - 2, real_y - 2, w + 4, h + 4)
                
    return best_grid

# --- Grid Line Detection ---
def find_grid_dividers(proj, min_gap=15, threshold_ratio=0.2):
    """Finds peaks in edge projection to identify rows/cols."""
    if len(proj) == 0: return []
    limit = np.max(proj) * threshold_ratio
    
    candidates = np.where(proj > limit)[0]
    if len(candidates) == 0: return []
    
    clusters = []
    current_cluster = [candidates[0]]
    
    for i in range(1, len(candidates)):
        if candidates[i] - candidates[i-1] < min_gap:
            current_cluster.append(candidates[i])
        else:
            clusters.append(int(np.mean(current_cluster)))
            current_cluster = [candidates[i]]
    clusters.append(int(np.mean(current_cluster)))
    
    return clusters

def cluster_coords(coords, min_dist=15):
    if not coords: return []
    coords = sorted(coords)
    clusters = []
    if not coords: return clusters
    curr = [coords[0]]
    for i in range(1, len(coords)):
        if coords[i] - curr[-1] < min_dist:
            curr.append(coords[i])
        else:
            clusters.append(int(np.mean(curr)))
            curr = [coords[i]]
    clusters.append(int(np.mean(curr)))
    return clusters

def extract_roi6(img_path, output_dir):
    img = cv2.imread(str(img_path))
    if img is None: 
        print(f"Skipping (Read Error): {img_path}")
        return
    
    print(f"Processing: {img_path.name}")
    
    # 1. Find the Main Grid (ROI-6)
    grid_rect = find_left_grid_anchor(img)
    
    if not grid_rect:
        print("  ! No grid anchor found. Skipping.")
        return

    gx, gy, gw, gh = grid_rect
    roi_img = img[gy:gy+gh, gx:gx+gw]
    
    # 2. Detect Rows and Columns inside the ROI
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Horizontal Projection (Rows) & Vertical Projection (Cols)
    proj_h = np.sum(edges, axis=1)
    proj_v = np.sum(edges, axis=0)
    
    # Hough Line Detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
    h_lines = []
    v_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):  # horizontal
                h_lines.append((y1 + y2) / 2)
            else:  # vertical
                v_lines.append((x1 + x2) / 2)
    
    rows = cluster_coords(h_lines, 25)
    cols = cluster_coords(v_lines, 20)
    
    # Fallback to projection if Hough sparse
    if len(rows) < 2 or len(cols) < 2:
        print("  Using projection fallback")
        rows = find_grid_dividers(proj_h, min_gap=15, threshold_ratio=0.2)
        cols = find_grid_dividers(proj_v, min_gap=15, threshold_ratio=0.2)
    
    row_coords = find_grid_dividers(proj_h, min_gap=20, threshold_ratio=0.15)
    col_coords = find_grid_dividers(proj_v, min_gap=20, threshold_ratio=0.15)
    
    # Ensure start/end points cover the edges
    if not row_coords or row_coords[0] > 10: row_coords = [0] + row_coords
    if row_coords[-1] < gh - 10: row_coords.append(gh)
    
    if not col_coords or col_coords[0] > 10: col_coords = [0] + col_coords
    if col_coords[-1] < gw - 10: col_coords.append(gw)
    
    # 3. Extract Thumbnails
    thumbnail_boxes = []
    
    for r in range(len(row_coords) - 1):
        for c in range(len(col_coords) - 1):
            y1, y2 = row_coords[r], row_coords[r+1]
            x1, x2 = col_coords[c], col_coords[c+1]
            
            w, h = x2 - x1, y2 - y1
            
            # Filter noise (too small to be a button)
            if w < 30 or h < 30: continue
            
            abs_x = gx + x1
            abs_y = gy + y1
            
            thumbnail_boxes.append({
                "bbox": [int(abs_x), int(abs_y), int(w), int(h)],
                "rel_bbox": [int(x1), int(y1), int(w), int(h)]
            })

    # 4. Detect Selection (Yellow Highlight)
    selected_index = -1
    max_yellow = 0
    hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    lower_yel = np.array([20, 100, 100])
    upper_yel = np.array([40, 255, 255])
    
    for i, item in enumerate(thumbnail_boxes):
        rx, ry, rw, rh = item["rel_bbox"]
        btn_roi = hsv_roi[ry:ry+rh, rx:rx+rw]
        mask = cv2.inRange(btn_roi, lower_yel, upper_yel)
        
        score = np.sum(mask > 0) / (rw * rh)
        if score > max_yellow:
            max_yellow = score
            selected_index = i

    # 5. Save Results
    safe_name = img_path.stem.replace('roi5_chart_tabs_', '') # Clean filename
    output_data = {
        "roi6_bbox": [int(gx), int(gy), int(gw), int(gh)],
        "thumbnails": [t["bbox"] for t in thumbnail_boxes],
        "selected_index": selected_index
    }
    
    # Save JSON
    json_path = output_dir / f"roi6_data_{safe_name}.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    # Save Visualization
    viz = img.copy()
    cv2.rectangle(viz, (gx, gy), (gx+gw, gy+gh), (0, 0, 255), 2) # Main Grid (Red)
    
    for i, item in enumerate(thumbnail_boxes):
        bx, by, bw, bh = item["bbox"]
        color = (255, 255, 0) # Cyan (Default)
        thickness = 1
        
        if i == selected_index:
            color = (0, 0, 255) # Red (Selected)
            thickness = 3
            
        cv2.rectangle(viz, (bx, by), (bx+bw, by+bh), color, thickness)
        cv2.putText(viz, str(i+1), (bx+5, by+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    viz_path = output_dir / f"viz_roi6_{safe_name}.png"
    cv2.imwrite(str(viz_path), viz)
    print(f"  ✓ Saved: {len(thumbnail_boxes)} thumbnails found. Selected: {selected_index+1}")

if __name__ == "__main__":
    roi5_dir = Path("ROI_5")
    roi6_dir = Path("ROI_6")
    roi6_dir.mkdir(exist_ok=True)
    
    # 1. Get all images from ROI_5
    all_files = sorted(list(roi5_dir.glob("*.png")) + list(roi5_dir.glob("*.jpg")))
    
    # 2. FILTER: IMPORTANT!
    # We want the FULL bottom-half images (roi0_bottom_half_test.png)
    # We do NOT want the small chart tabs (roi5_chart_tabs...) or viz files
    inputs = [f for f in all_files if "roi5_chart_tabs" not in f.name and "viz_" not in f.name]
    
    if not inputs:
        print("No valid input images found in ROI_5.")
        print("Please ensure 'roi0_bottom_half_test.png' (or similar) is in the ROI_5 folder.")
    else:
        for f in inputs:
            extract_roi6(f, roi6_dir)