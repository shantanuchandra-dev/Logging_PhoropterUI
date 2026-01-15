#!/ env python3
"""
Extract ROI-6 Chart Options Grid from screenshots.
Uses Hough Line Transform for grid detection and identifies the selected thumbnail.
Robust to different UI states and avoids labels/footer.
"""

import cv2
import numpy as np
import json
from pathlib import Path

def find_all_lines(proj, threshold_ratio=0.15, min_gap=20):
    if len(proj) == 0: return []
    limit = np.max(proj) * threshold_ratio
    peaks = []
    temp_proj = proj.copy()
    while True:
        idx = np.argmax(temp_proj)
        if temp_proj[idx] < limit: break
        peaks.append(idx)
        start = max(0, idx - min_gap)
        end = min(len(temp_proj), idx + min_gap)
        temp_proj[start:end] = 0
    return sorted(peaks)

def cluster_coords(coords, min_dist=15):
    if not coords: return []
    coords = sorted(coords)
    clusters = []
    if not coords: return []
    curr = [coords[0]]
    for i in range(1, len(coords)):
        if coords[i] - coords[i-1] < min_dist: curr.append(coords[i])
        else:
            clusters.append(int(np.mean(curr)))
            curr = [coords[i]]
    clusters.append(int(np.mean(curr)))
    return clusters

def find_roi5_strip(img):
    """Reuse the template matching logic from ROI-5 to find the anchor strip."""
    template_path = Path("ROI_5/chart_template.png")
    if not template_path.exists():
        return None
    
    template = cv2.imread(str(template_path))
    height, width = img.shape[:2]
    
    search_template = template
    if width < 700:
        scale = width / 929.0
        search_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
    
    tw, th = search_template.shape[1], search_template.shape[0]
    res = cv2.matchTemplate(img, search_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.6)
    matches = list(zip(*loc[::-1]))
    
    if not matches:
        return None
        
    y_coords = [pt[1] for pt in matches]
    y_level = max(set(y_coords), key=y_coords.count)
    min_x = min(pt[0] for pt in matches)
    max_x = max(pt[0] for pt in matches)
    
    roi5_y_end = y_level + th + 10
    
    return {
        "y_end": roi5_y_end,
        "x_start": max(0, min_x - 10),
        "x_end": min(width, max_x + tw + 100)
    }

def is_slot_chart(roi):
    if roi.size == 0: return False
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std = np.std(gray_roi)
    white_ratio = np.sum(gray_roi > 230) / roi.size
    # Valid chart buttons are either high pattern variance OR white backgrounds
    # Increased std threshold to 15.0 to avoid gold background noise
    if std > 15.0: return True
    if white_ratio > 0.05: return True
    return False

def extract_roi6(img_path, output_dir):
    img = cv2.imread(str(img_path))
    if img is None: return
    
    height, width = img.shape[:2]
    roi5 = find_roi5_strip(img)
    if not roi5: return

    # --- Container Search Area ---
    y_search_start = roi5["y_end"]
    x_search_start = roi5["x_start"]
    x_search_end = roi5["x_end"]
    
    # "within the first 60% of the ROI0" interpretation here: 
    # search the chart region primarily, which is top part of bottom half
    remaining_h = height - y_search_start
    y_search_end = min(height, y_search_start + int(remaining_h * 0.9))
    
    roi6_search_area = img[y_search_start:y_search_end, x_search_start:x_search_end]
    
    # --- Detect Yellow Panel ---
    lower_yellow = np.array([5, 20, 20]) 
    upper_yellow = np.array([60, 255, 255])
    hsv_search = cv2.cvtColor(roi6_search_area, cv2.COLOR_BGR2HSV)
    panel_mask = cv2.inRange(hsv_search, lower_yellow, upper_yellow)
    kernel = np.ones((5, 5), np.uint8)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(panel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    p_bbox = [0, 0, roi6_search_area.shape[1], roi6_search_area.shape[0]]
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 2000:
            p_bbox = cv2.boundingRect(cnt)
            
    px, py, pw, ph = p_bbox
    roi6_strip_raw = roi6_search_area[py:py+ph, px:px+pw]
    
    # --- Hough Line detection within Panel ---
    gray = cv2.cvtColor(roi6_strip_raw, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(thresh, 50, 150)
    
    # HoughLinesP to identify exactly the 7x3 dividers
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=ph//4, maxLineGap=20)
    
    h_lines = []
    v_lines = []
    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            if abs(ly1 - ly2) < 5: h_lines.append((ly1 + ly2) // 2)
            elif abs(lx1 - lx2) < 5: v_lines.append((lx1 + lx2) // 2)

    rows = cluster_coords(h_lines, 20)
    cols = cluster_coords(v_lines, 30)
    
    # Fallback to projection if Hough is sparse
    if len(rows) < 2 or len(cols) < 2:
        proj_h = np.sum(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3), axis=1)
        rows = find_all_lines(np.abs(proj_h), 0.15, int(ph/5))
        cols = [int(i * pw / 7) for i in range(8)]

    if not rows or rows[0] > 10: rows = [0] + rows
    if not rows or rows[-1] < ph - 10: rows.append(ph)
    if not cols or cols[0] > 10: cols = [0] + cols
    if not cols or cols[-1] < pw - 10: cols.append(pw)

    thumbnail_boxes = []
    for r in range(len(rows) - 1):
        for c in range(len(cols) - 1):
            by1, by1_end = rows[r], rows[r+1]
            bx1, bx2 = cols[c], cols[c+1]
            tw, th = bx2 - bx1, by1_end - by1
            if tw < 30 or th < 25: continue
            
            roi = roi6_strip_raw[by1:by1_end, bx1:bx2]
            if is_slot_chart(roi):
                abs_x = x_search_start + px + bx1
                abs_y = y_search_start + py + by1
                thumbnail_boxes.append([int(abs_x), int(abs_y), int(tw), int(th)])

    thumbnail_boxes = sorted(thumbnail_boxes, key=lambda b: (b[1], b[0]))
    if not thumbnail_boxes: return

    # --- Identify Selected ---
    selected_index = -1
    max_yellow = 0
    full_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_sel = np.array([10, 80, 80])
    upper_sel = np.array([40, 255, 255])

    for i, box in enumerate(thumbnail_boxes):
        ax, ay, aw, ah = box
        roi = full_hsv[ay:ay+ah, ax:ax+aw]
        if roi.size == 0: continue
        mask = cv2.inRange(roi, lower_sel, upper_sel)
        score = np.sum(mask > 0) / (aw * ah)
        if score > max_yellow:
            max_yellow = score
            selected_index = i

    # --- Prepare Output ---
    all_x1 = [b[0] for b in thumbnail_boxes]
    all_y1 = [b[1] for b in thumbnail_boxes]
    all_x2 = [b[0] + b[2] for b in thumbnail_boxes]
    all_y2 = [b[1] + b[3] for b in thumbnail_boxes]
    tx1, ty1, tx2, ty2 = min(all_x1), min(all_y1), max(all_x2), max(all_y2)

    safe_name = img_path.stem.replace(' ', '_').replace(' ', '_')
    output_data = {
        "roi6_bbox": [int(tx1), int(ty1), int(tx2 - tx1), int(ty2 - ty1)],
        "thumbnails": [{"bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])]} for b in thumbnail_boxes],
        "selected_index": int(selected_index),
    }
    if selected_index != -1:
        sel = thumbnail_boxes[selected_index]
        output_data["selected_bbox"] = [int(sel[0]), int(sel[1]), int(sel[2]), int(sel[3])]

    json_path = output_dir / f"roi6_data_{safe_name}.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # --- Viz ---
    viz = img.copy()
    cv2.rectangle(viz, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
    for i, box in enumerate(thumbnail_boxes):
        bx, by, bw, bh = box
        color = (255, 0, 0)
        if i == selected_index: color = (0, 255, 255)
        cv2.rectangle(viz, (bx, by), (bx + bw, by + bh), color, 2)
        cv2.putText(viz, str(i), (bx + 2, by + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    viz_path = output_dir / f"viz_roi6_{safe_name}.png"
    cv2.imwrite(str(viz_path), viz)
    print(f"✓ {img_path.name}: {len(thumbnail_boxes)} thumbs, selected={selected_index}")

if __name__ == "__main__":
    roi6_dir = Path("ROI_6")
    roi6_dir.mkdir(exist_ok=True)
    roi5_dir = Path("ROI_5")
    inputs = []
    for ext in ['*.png', '*.webp', '*.jpg']:
        inputs.extend(list(roi5_dir.glob(ext)))
    exclude = ('roi5_chart_tabs', 'viz_blocks', 'viz_dynamic', 'dynamic_roi5', 'test_refine', 'viz_refine', 'crop_test', 'viz_test', 'chart_template')
    inputs = [f for f in inputs if not f.name.lower().startswith(exclude)]
    for img_file in sorted(inputs):
        extract_roi6(img_file, roi6_dir)
