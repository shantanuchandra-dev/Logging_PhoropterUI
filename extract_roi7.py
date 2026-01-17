#!/env python3
"""
Extract ROI-7 Big Chart Pane from screenshots.
Uses dynamic contour detection to isolate the large white/black pane on the right.
Correlates identity with the selected thumbnail in ROI-6.
"""

import cv2
import numpy as np
import json
from pathlib import Path

def find_roi7_pane(img, color_hint=None):
    """Detect the big chart pane by merging adjacent rectangular parts."""
    height, width = img.shape[:2]
    
    # 1. Search zone
    search_x = int(width * 0.58) 
    search_y1 = int(height * 0.15)
    search_y2 = int(height * 0.95)
    right_zone = img[search_y1:search_y2, search_x:]
    
    gray = cv2.cvtColor(right_zone, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # 2. Find all significant rectangular candidates
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (width * height * 0.015): continue
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(approx)
            if 0.3 < (w/h) < 3.0:
                rects.append([x, y, w, h])

    if not rects: return None
    
    # 3. Merge vertically stacked rects (for two-tone charts)
    rects.sort(key=lambda b: b[1]) # Sort by Y
    merged = []
    for r in rects:
        if not merged:
            merged.append(r)
            continue
        
        last = merged[-1]
        # If this rect is just below the last one and has VERY similar width/X
        y_gap = r[1] - (last[1] + last[3])
        x_diff = abs(r[0] - last[0])
        w_diff = abs(r[2] - last[2])
        
        # Slightly more generous alignment for different scales
        if y_gap < 25 and x_diff < 10 and w_diff < 10:
            # Merge
            new_r = [
                min(last[0], r[0]),
                last[1],
                max(last[2], r[2]),
                (r[1] + r[3]) - last[1]
            ]
            merged[-1] = new_r
        else:
            merged.append(r)
            
    # 4. Pick the best merged candidate (rightmost large one)
    candidates = []
    for m in merged:
        mx, my, mw, mh = m
        area = mw * mh
        # Filter for typical chart sizes
        if area < (width * height * 0.03): continue
        if area > (width * height * 0.15): continue
        
        rightness = (mx + mw/2) / (width - search_x)
        score = area * rightness
        candidates.append((score, [mx + search_x, my + search_y1, mw, mh]))
        
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
        
    return None
        
    # Fallback to simple projection if no rectangle found
    _, mask = cv2.threshold(gray, 235 if color_hint == 'white' else 35, 255, 
                              cv2.THRESH_BINARY if color_hint == 'white' else cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        best_cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        x1, y1, w1, h1 = cv2.boundingRect(best_cnt)
        return [int(x1 + search_x), int(y1 + search_y1), int(w1), int(h1)]
        
    return None

def get_color_hint(img, roi6_data):
    """Determine if the selected chart thumbnail is mostly black or white."""
    idx = roi6_data.get("selected_index", -1)
    thumbs = roi6_data.get("thumbnails", [])
    if idx < 0 or idx >= len(thumbs): return None
    
    bx, by, bw, bh = thumbs[idx]["bbox"]
    # Check if this is the bottom half image (ROI_5 samples are already cropped)
    # We'll assume the thumbnail coords are relative to ROI_0 (full bottom half)
    thumb_crop = img[by:by+bh, bx:bx+bw]
    if thumb_crop.size == 0: return None
    
    gray = cv2.cvtColor(thumb_crop, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    return "white" if mean_val > 128 else "black"

def extract_roi7(img_path, output_dir):
    img = cv2.imread(str(img_path))
    if img is None: return
    
    # 1. Get Context from ROI-6
    safe_name = img_path.stem.replace(' ', '_').replace(' ', '_')
    roi6_json_path = Path("ROI_6") / f"roi6_data_{safe_name}.json"
    
    color_hint = None
    chart_info = {"identity": "Unknown", "matches_roi6": False}
    
    if roi6_json_path.exists():
        with open(roi6_json_path, 'r') as f:
            roi6_data = json.load(f)
            color_hint = get_color_hint(img, roi6_data)
            chart_info["roi6_selected_index"] = roi6_data.get("selected_index", -1)
            chart_info["matches_roi6"] = True
            chart_info["color_hint"] = color_hint
            
    # 2. Detection with Hint
    roi7_bbox = find_roi7_pane(img, color_hint)
    if not roi7_bbox:
        print(f"Warning: Could not detect ROI-7 pane in {img_path.name} (hint={color_hint})")
        return
        
    x, y, w, h = roi7_bbox
    roi7_crop = img[y:y+h, x:x+w]
    
    # 3. Output
    output_data = {
        "roi7_bbox": [int(x), int(y), int(w), int(h)],
        "chart_info": chart_info
    }
    
    json_path = output_dir / f"roi7_data_{safe_name}.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    # Save crop
    crop_path = output_dir / f"roi7_chart_{safe_name}.png"
    cv2.imwrite(str(crop_path), roi7_crop)
    
    # 4. Viz
    viz = img.copy()
    cv2.rectangle(viz, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(viz, "ROI-7 BIG CHART", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    viz_path = output_dir / f"viz_roi7_{safe_name}.png"
    cv2.imwrite(str(viz_path), viz)
    
    # Debug edges
    # edges_path = output_dir / f"debug_edges_{safe_name}.png"
    # cv2.imwrite(str(edges_path), find_roi7_pane.debug_edges)
    
    print(f"✓ {img_path.name}: ROI-7 extracted at {roi7_bbox}, correlated with ROI-6 idx {chart_info.get('roi6_selected_index')}")

if __name__ == "__main__":
    roi7_dir = Path("ROI_7")
    roi7_dir.mkdir(exist_ok=True)
    
    # We use ROI_5 as the source of samples (bottom halves)
    roi5_dir = Path("ROI_5")
    inputs = []
    for ext in ['*.png', '*.webp', '*.jpg']:
        inputs.extend(list(roi5_dir.glob(ext)))
             
    exclude = ('roi5_chart_tabs', 'viz_blocks', 'viz_dynamic', 'dynamic_roi5', 'test_refine', 'viz_refine', 'crop_test', 'viz_test', 'chart_template')
    inputs = [f for f in inputs if not f.name.lower().startswith(exclude)]
    
    for img_file in sorted(inputs):
        extract_roi7(img_file, roi7_dir)
