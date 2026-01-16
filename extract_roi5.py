#!/usr/bin/env python3
"""
Extract ROI-5 Chart tabs region from bottom half images
"""

import cv2
import numpy as np
from pathlib import Path
import json

def extract_chart_tabs(input_path, output_dir):
    """
    Extract the horizontal strip containing Chart1-5 tabs using dynamic detection
    """
    
    # Read the image
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Error reading {input_path}")
        return
    
    height, width = img.shape[:2]
    
    # --- Dynamic Detection Logic ---
    template_path = Path("ROI_5/chart_template.png")
    if not template_path.exists():
        print("Warning: Template not found. Using dynamic relative coordinates.")
        h, w = img.shape[:2]
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
        res = cv2.matchTemplate(img, search_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.3)
        matches = list(zip(*loc[::-1]))
        
        if not matches:
             print("Warning: No Chart labels detected. Using fallback.")
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
    
    print(f"Extraction region: y={y_start}-{y_end}, x={x_start}-{x_end}")
    chart_tabs = img[y_start:y_end, x_start:x_end]
    
    safe_name = Path(input_path).stem.replace('roi0_bottom_half_', '').replace(' ', '_').replace(' ', '_')
    output_filename = f"roi5_chart_tabs_{safe_name}.png"
    output_path = Path(output_dir) / output_filename
    
    cv2.imwrite(str(output_path), chart_tabs)
    print(f"✓ Created: {output_path}")
    
    visualize_tab_blocks(img, (y_start, y_end, x_start, x_end), output_dir, safe_name, boundaries)
    detect_selected_tab(chart_tabs, output_path, boundaries)
    
    return output_path

def visualize_tab_blocks(full_img, coords, output_dir, timestamp, boundaries):
    """
    Draw bounding boxes for each of the 5 detected chart blocks using dynamic boundaries
    """
    y1, y2, x1, x2 = coords
    viz = full_img.copy()
    
    for i in range(len(boundaries) - 1):
        tx1 = boundaries[i]
        tx2 = boundaries[i+1]
        
        # Draw box for this tab (Cyan)
        cv2.rectangle(viz, (tx1, y1), (tx2, y2), (255, 255, 0), 2)
        cv2.putText(viz, f"C{i+1}", (tx1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    viz_filename = f"viz_blocks_{timestamp}.png"
    viz_path = Path(output_dir) / viz_filename
    cv2.imwrite(str(viz_path), viz)
    print(f"✓ Created Block Viz: {viz_path}")

def detect_selected_tab(chart_tabs_img, output_path, boundaries):
    """
    Detect which chart tab is selected using dynamic boundaries
    """
    selected_index = None
    max_yellow_score = 0
    
    print("\n  Analyzing tabs:")
    for i in range(len(boundaries) - 1):
        # Get region for this tab
        bx1 = boundaries[i] - boundaries[0]  # Relative to crop
        bx2 = boundaries[i+1] - boundaries[0]
        tab_region = chart_tabs_img[:, bx1:bx2]
        
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
        
        print(f"    Tab {i+1}: yellow_score = {yellow_score:.3f} (x={bx1}-{bx2})")
        
        if yellow_score > max_yellow_score:
            max_yellow_score = yellow_score
            selected_index = i
    
    if selected_index is not None:
        print(f"\n  → Selected tab: Chart{selected_index + 1} (index={selected_index})")
        
        # Save result as text file
        result_file = str(output_path).replace('.png', '_selected.txt')
        with open(result_file, 'w') as f:
            f.write(f"selected_chart_tab: Chart{selected_index + 1}\nindex: {selected_index}\nconfidence: {max_yellow_score:.3f}")
        
        # Save result as JSON file
        json_file = str(output_path).replace('.png', '_selected.json')
        with open(json_file, 'w') as f:
            json.dump({
                "selected_chart_tab": f"Chart{selected_index + 1}",
                "index": selected_index,
                "confidence": round(max_yellow_score, 3)
            }, f, indent=2)
        print(f"  ✓ Saved JSON to: {json_file}")
    
    # Optional: Try OCR on the tabs
    try_ocr_tabs(chart_tabs_img)

def try_ocr_tabs(chart_tabs_img):
    """
    Attempt OCR to read tab labels
    """
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(chart_tabs_img)
        
        if results:
            print("\n  OCR Results:")
            for (bbox, text, confidence) in results:
                print(f"    '{text}' (confidence: {confidence:.2f})")
    except ImportError:
        print("\n  OCR skipped (easyocr not installed)")
    except Exception as e:
        print(f"\n  OCR error: {e}")

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
        extract_chart_tabs(img_file, roi5_dir)
        print("-" * 30)
    
    print("✓ All chart tabs extracted!")