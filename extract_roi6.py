#!/usr/bin/env python3
"""
Extract ROI-6 Chart Options Grid
"""

import cv2
import numpy as np
import json
from pathlib import Path
import os
import datetime


def extract(roi0_img, save_debug=False, filename=None, output_dir='ROI_6'):
    """
    Extract chart options grid (ROI-6) from ROI-0 image array.
    
    Args:
        roi0_img: ROI-0 image as numpy array
        save_debug: Whether to save debug images
        filename: Original filename for debug output naming
        output_dir: Directory to save debug images
    
    Returns:
        dict: {
            'roi_id': 'ROI_6',
            'bbox': [x, y, w, h],  # Overall grid bbox
        }
    """
    grid_rect = _find_left_grid_anchor(roi0_img)
    
    if not grid_rect:
        return {
            'roi_id': 'ROI_6',
            'bbox': [],
            'thumbnails': [],
            'selected_index': -1,
            'error': 'No grid anchor found'
        }

    gx, gy, gw, gh = grid_rect
    roi_img = roi0_img[gy:gy+gh, gx:gx+gw]
    
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
    
    rows = _cluster_coords(h_lines, 25)
    cols = _cluster_coords(v_lines, 20)
    
    # Fallback to projection if Hough sparse
    if len(rows) < 2 or len(cols) < 2:
        rows = _find_grid_dividers(proj_h, min_gap=20, threshold_ratio=0.15)
        cols = _find_grid_dividers(proj_v, min_gap=20, threshold_ratio=0.15)
    
    row_coords = _find_grid_dividers(proj_h, min_gap=20, threshold_ratio=0.15)
    col_coords = _find_grid_dividers(proj_v, min_gap=20, threshold_ratio=0.15)
    
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
            
            thumbnail_boxes.append([int(abs_x), int(abs_y), int(w), int(h)])

    # 4. Detect Selection (Yellow Highlight)
    selected_index = -1
    max_yellow = 0
    hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    lower_yel = np.array([20, 100, 100])
    upper_yel = np.array([40, 255, 255])
    
    for i, bbox in enumerate(thumbnail_boxes):
        abs_x, abs_y, w, h = bbox
        rx = abs_x - gx
        ry = abs_y - gy
        btn_roi = hsv_roi[ry:ry+h, rx:rx+w]
        mask = cv2.inRange(btn_roi, lower_yel, upper_yel)
        
        score = np.sum(mask > 0) / (w * h) if (w * h) > 0 else 0
        if score > max_yellow:
            max_yellow = score
            selected_index = i

    result = {
        'roi_id': 'ROI_6',
        'bbox': [int(gx), int(gy), int(gw), int(gh)],
        'thumbnails': thumbnail_boxes,
        'selected_index': selected_index
    }
    if save_debug:
        os.makedirs(output_dir, exist_ok=True)
        now = datetime.datetime.now().strftime('%d%m_%H%M%S')
        if filename is not None:
            input_base = os.path.splitext(os.path.basename(filename))[0]
            prefix = input_base[:4]
        else:
            prefix = 'roi6'
        # Save Visualization
        viz = roi0_img.copy()
        cv2.rectangle(viz, (gx, gy), (gx+gw, gy+gh), (0, 0, 255), 2) # Main Grid (Red)
        for i, bbox in enumerate(thumbnail_boxes):
            bx, by, bw, bh = bbox
            color = (0, 0, 255) if i == selected_index else (255, 255, 0)  # Red if selected, cyan otherwise
            thickness = 3 if i == selected_index else 1
            cv2.rectangle(viz, (bx, by), (bx+bw, by+bh), color, thickness)
            cv2.putText(viz, str(i+1), (bx+5, by+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        viz_path = os.path.join(output_dir, f'{prefix}_{now}_grid_debug.png')
        cv2.imwrite(viz_path, viz)
        result['image_path'] = viz_path
    return result


def _find_left_grid_anchor(img):
    # Finds the chart grid in the bottom-left area.
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


def _find_grid_dividers(proj, min_gap=15, threshold_ratio=0.2):
    # Finds peaks in edge projection to identify rows/cols.
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


def _cluster_coords(coords, min_dist=15):
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
            print(f"Processing: {f.name}")
            img = cv2.imread(str(f))
            if img is not None:
                result = extract(str(f), save_debug=True)
                print(f"  Result: {result}")