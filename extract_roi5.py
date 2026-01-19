#!/usr/bin/env python3
"""
Extract ROI-5 Chart tabs region from bottom half images
"""

import cv2
import numpy as np
from pathlib import Path
import json
import os


def extract(roi0_img, save_debug=False, output_dir='ROI_5'):
    """
    Extract chart tabs (ROI-5) from ROI-0 image.
    
    Args:
        roi0_img: ROI-0 image (numpy array)
        save_debug: Whether to save debug images
        output_dir: Directory to save debug images
    
    Returns:
        dict: {
            'roi_id': 'ROI_5',
            'bbox': [x, y, w, h],  # Overall tabs region bbox
            'tab_boundaries': [x1, x2, x3, x4, x5, x6],  # X coordinates of tab dividers
            'selected_tab': int,  # Index of selected tab (0-4)
            'confidence': float,  # Confidence score for selection
            'image_path': 'path/to/debug_image.png' (if save_debug=True)
        }
    """
    height, width = roi0_img.shape[:2]
    
    # --- Dynamic Detection Logic ---
    template_path = Path("ROI_5/chart_template.png")
    if not template_path.exists():
        h, w = roi0_img.shape[:2]
        y_start = int(h * 0.22)
        y_end = int(h * 0.31)
        x_start = int(w * 0.175)
        x_end = int(w * 0.47)
        tab_width = (x_end - x_start) / 5
        # Adjust for wider Tab1 to avoid overlap
        boundaries = [x_start, x_start + 120, x_start + 120 + 88, x_start + 120 + 176, x_start + 120 + 264, x_end]
    else:
        template = cv2.imread(str(template_path))
        search_template = template
        if width < 700:
            scale = width / 929.0
            search_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        
        tw, th = search_template.shape[1], search_template.shape[0]
        res = cv2.matchTemplate(roi0_img, search_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.3)
        matches = list(zip(*loc[::-1]))
        
        if not matches:
             y_start, y_end, x_start, x_end = 0, 50, 0, 500
             boundaries = [0, 100, 200, 300, 400, 500]
        else:
            # Group matches into 5 tab anchors
            y_coords = [pt[1] for pt in matches]
            y_level = max(set(y_coords), key=y_coords.count)
            min_x = min(pt[0] for pt in matches)
            max_x = max(pt[0] for pt in matches)
            
            y_start, y_end = y_level, y_level + th  # Use template height
            x_start, x_end = min_x, max_x + tw  # No extra padding
            
            # Calculate boundaries for 5 tabs
            tab_width = (x_end - x_start) / 5
            boundaries = [int(x_start + i * tab_width) for i in range(6)]
    
    # Boundary safety check
    y_start = min(max(0, y_start), height - 1)
    y_end = min(max(y_start + 1, y_end), height)
    x_start = min(max(0, x_start), width - 1)
    x_end = min(max(x_start + 1, x_end), width)
    
    chart_tabs = roi0_img[y_start:y_end, x_start:x_end]
    
    # Detect selected tab
    selected_index = None
    max_yellow_score = 0
    
    for i in range(len(boundaries) - 1):
        # Get region for this tab
        bx1 = boundaries[i] - x_start  # Relative to crop
        bx2 = boundaries[i+1] - x_start
        tab_region = chart_tabs[:, bx1:bx2]
        
        if tab_region.size == 0:
            continue
        
        # Convert to HSV to detect yellow/orange color
        hsv = cv2.cvtColor(tab_region, cv2.COLOR_BGR2HSV)
        
        # Yellow/orange range in HSV
        lower_yellow = np.array([10, 80, 80])
        upper_yellow = np.array([40, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_pixels = np.sum(mask > 0)
        yellow_score = yellow_pixels / (tab_region.shape[0] * tab_region.shape[1])
        
        if yellow_score > max_yellow_score:
            max_yellow_score = yellow_score
            selected_index = i
    
    result = {
        'roi_id': 'ROI_5',
        'bbox': [int(x_start), int(y_start), int(x_end - x_start), int(y_end - y_start)],
        'tab_boundaries': [int(b) for b in boundaries],
        'selected_tab': selected_index if selected_index is not None else -1,
        'confidence': float(max_yellow_score)
    }
    
    if save_debug:
        os.makedirs(output_dir, exist_ok=True)
        import datetime
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save chart tabs image
        tabs_path = os.path.join(output_dir, f'roi5_chart_tabs_{now}.png')
        cv2.imwrite(tabs_path, chart_tabs)
        
        # Save visualization with boxes
        viz = roi0_img.copy()
        for i in range(len(boundaries) - 1):
            tx1 = boundaries[i]
            tx2 = boundaries[i+1]
            color = (0, 0, 255) if i == selected_index else (255, 255, 0)
            thickness = 3 if i == selected_index else 2
            cv2.rectangle(viz, (tx1, y_start), (tx2, y_end), color, thickness)
            cv2.putText(viz, f"C{i+1}", (tx1 + 5, y_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        viz_path = os.path.join(output_dir, f'viz_roi5_{now}.png')
        cv2.imwrite(viz_path, viz)
        
        result['image_path'] = tabs_path
    
    return result


if __name__ == "__main__":
    # Process all relevant images in ROI_5
    roi5_dir = Path("ROI_5")
    # Include standard bottom halves, sample webp images, and other common formats
    files_to_process = []
    for ext in ['*.png', '*.webp', '*.jpg']:
        files_to_process.extend(list(roi5_dir.glob(ext)))
    
    # Filter for files that are inputs (not products of this script)
    exclude_prefixes = ('roi5_chart_tabs', 'viz_blocks', 'viz_dynamic', 'dynamic_roi5', 'test_refine', 'viz_refine', 'crop_test', 'viz_test', 'chart_template')
    inputs = [f for f in files_to_process if not f.name.lower().startswith(exclude_prefixes)]
    inputs = sorted(inputs, key=lambda x: x.name)
    
    print(f"Found {len(inputs)} input images to process\n")
    
    for img_file in inputs:
        print(f"Processing: {img_file.name}")
        img = cv2.imread(str(img_file))
        if img is not None:
            result = extract(img, save_debug=True)
            print(f"  Result: {result}")
        print("-" * 30)
    
    print("✓ All chart tabs extracted!")