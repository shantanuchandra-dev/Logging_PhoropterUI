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
    """Detect the big chart pane by finding the most prominent black rectangular border."""
    height, width = img.shape[:2]
    # Search right half, avoiding the very top/bottom and central SPH table
    search_x = int(width * 0.58) 
    search_y1 = int(height * 0.15) 
    search_y2 = int(height * 0.95)
    right_zone = img[search_y1:search_y2, search_x:]
    
    gray = cv2.cvtColor(right_zone, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Detect edges to find the black border lines
    edges = cv2.Canny(blurred, 30, 100)
    
    # 2. Find rectangular contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (width * height * 0.03): continue
        
        # Approximate to find rectangles
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h
            # Typical chart pane aspect ratio
            if 0.6 < aspect < 2.0:
                # Color check: does the interior match the color_hint?
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [approx], -1, 255, -1)
                mean_int = cv2.mean(gray, mask=mask)[0]
                
                score = area
                if color_hint == 'white' and mean_int > 200: score *= 2
                elif color_hint == 'black' and mean_int < 60: score *= 2
                elif color_hint and ((color_hint == 'white' and mean_int < 150) or (color_hint == 'black' and mean_int > 150)):
                    continue # Color mismatch
                
                # Favor candidates further to the right to avoid SPH table
                right_bias = (x + w) / right_zone.shape[1]
                candidates.append((score * right_bias, [x + search_x, y + search_y1, w, h]))
                
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
        
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
