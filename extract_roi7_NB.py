#!/usr/bin/env python3
"""
Extract ROI-7 Big Chart Pane from screenshots.
Uses dynamic contour detection to isolate the large white/black pane on the right.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import os
import datetime


def extract(roi0_img, roi6_data=None, save_debug=False, filename=None, output_dir='ROI_7'):
    """
    Extract big chart pane (ROI-7) from ROI-0 image array.
    
    Args:
        roi0_img: ROI-0 image as numpy array
        roi6_data: Optional ROI-6 data dict for color hint
        save_debug: Whether to save debug images
        filename: Original filename for debug output naming
        output_dir: Directory to save debug images
    
    Returns:
        dict: {
            'roi_id': 'ROI_7',
            'bbox': [x, y, w, h],  # Chart pane bbox
            'chart_info': {
                'color_hint': 'white' or 'black',
                'matches_roi6': bool,
                'roi6_selected_index': int
            },
            'image_path': 'path/to/debug_image.png' (if save_debug=True)
        }
    """
    if roi0_img is None:
        raise ValueError('Input image (roi0_img) is None')
    # 1. Get color hint from ROI-6 if available
    color_hint = None
    chart_info = {"identity": "Unknown", "matches_roi6": False}
    if roi6_data:
        color_hint = _get_color_hint(roi0_img, roi6_data)
        chart_info["roi6_selected_index"] = roi6_data.get("selected_index", -1)
        chart_info["matches_roi6"] = True
        chart_info["color_hint"] = color_hint
    # 2. Detection with Hint
    roi7_bbox = _find_roi7_pane(roi0_img, color_hint)
    if not roi7_bbox:
        return {
            'roi_id': 'ROI_7',
            'bbox': [],
            'chart_info': chart_info,
            'error': f'Could not detect ROI-7 pane (hint={color_hint})'
        }
    x, y, w, h = roi7_bbox
    roi7_crop = roi0_img[y:y+h, x:x+w]

    # Perform OCR/classification
    ocr_result = _perform_ocr(roi7_crop)
    chart_info.update(ocr_result)

    result = {
        'roi_id': 'ROI_7',
        'bbox': [int(x), int(y), int(w), int(h)],
        'chart_info': chart_info
    }
    if save_debug:
        os.makedirs(output_dir, exist_ok=True)
        now = datetime.datetime.now().strftime('%d%m_%H%M%S')
        if filename is not None:
            input_base = os.path.splitext(os.path.basename(filename))[0]
            prefix = input_base[:4]
        else:
            prefix = 'img'
        # Save crop
        crop_path = os.path.join(output_dir, f'{prefix}_{now}_roi7_debug.png')
        cv2.imwrite(crop_path, roi7_crop)
        # Save Viz
        viz = roi0_img.copy()
        cv2.rectangle(viz, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(viz, "ROI-7 BIG CHART", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        viz_path = os.path.join(output_dir, f'{prefix}_{now}_viz.png')
        cv2.imwrite(viz_path, viz)
        result['image_path'] = crop_path
    return result


def _find_roi7_pane(img, color_hint=None):
    """Detect the big chart pane by merging adjacent rectangular parts."""
    height, width = img.shape[:2]
    
    # 1. Search zone
    search_x = int(width * 0.60)
    search_y1 = int(height * 0.60)
    search_y2 = height
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


def _get_color_hint(img, roi6_data):
    """Determine if the selected chart thumbnail is mostly black or white."""
    idx = roi6_data.get("selected_index", -1)
    thumbs = roi6_data.get("thumbnails", [])
    if idx < 0 or idx >= len(thumbs): return None
    
    bx, by, bw, bh = thumbs[idx]
    # Check if this is the bottom half image (ROI_5 samples are already cropped)
    # We'll assume the thumbnail coords are relative to ROI_0 (full bottom half)
    thumb_crop = img[by:by+bh, bx:bx+bw]
    if thumb_crop.size == 0: return None
    
    gray = cv2.cvtColor(thumb_crop, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    return "white" if mean_val > 128 else "black"


def _perform_ocr(roi7_crop):
    """Placeholder for OCR on the cropped ROI-7 chart pane."""
    # TODO: Replace with actual OCR/classification logic (e.g., EasyOCR or a custom model)
    # For now, we return a hardcoded placeholder until a specific OCR library is integrated.
    # To avoid dependency issues, we leave the implementation to the user to fill in or approve
    # a specific library integration.
    return {
        "chart_symbol": "M (Placeholder)",
        "confidence": 0.01 # Low confidence for placeholder
    }


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
        print(f"Processing: {img_file.name}")
        img = cv2.imread(str(img_file))
        if img is not None:
            # Try to load ROI-6 data for color hint
            roi6_json_path = Path("ROI_6") / f"roi6_data_{img_file.stem}.json"
            roi6_data = None
            if roi6_json_path.exists():
                with open(roi6_json_path, 'r') as f:
                    roi6_data = json.load(f)
            
            result = extract(str(img_file), roi6_data=roi6_data, save_debug=True)
            print(f"  Result: {result}")
