#!/usr/bin/env python3
"""
Extract ROI-5 Chart tabs region - Auto-Detect Version
Works on both Full Screenshots and Cropped Bottom-Halves
"""

import cv2
import numpy as np
from pathlib import Path

def detect_tab_strip_smart(img):
    """
    Scans the image to find the row of 5 Chart Tabs (Chart1 - Chart5)
    based on their shape and location, not hardcoded numbers.
    """
    h, w = img.shape[:2]
    
    # 1. Define Search Zone (Bottom-Left Quadrant)
    # The tabs are always in the bottom half of the screen, on the left side.
    # If it's a full screenshot (h > 800), look in the lower half.
    # If it's already cropped (h < 600), look in the top half of that crop.
    
    if h > 800: # Full HD Screenshot
        y_min, y_max = int(h * 0.55), int(h * 0.75)
    else:       # Already cropped bottom half
        y_min, y_max = int(h * 0.10), int(h * 0.50)
        
    x_max = int(w * 0.6) # Tabs are on the left side
    
    # Crop to search zone to avoid noise
    search_roi = img[y_min:y_max, 0:x_max]
    
    # 2. Find Horizontal Structures
    gray = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
    
    # Use gradient to find edges of buttons
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # Threshold to get strong edges
    _, binary = cv2.threshold(grad, 40, 255, cv2.THRESH_BINARY)
    
    # "Smear" horizontally to connect the 5 buttons into one long strip
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 3. Find Contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_box = None
    max_score = 0
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        # Filter: The strip of 5 buttons is wide and thin.
        aspect_ratio = cw / float(ch)
        
        # Valid Strip Criteria:
        # - Aspect ratio > 6 (it's very wide)
        # - Width > 200px (it's big)
        # - Height between 20px and 80px (standard button height)
        if aspect_ratio > 6 and cw > 200 and 20 < ch < 80:
            score = cw * ch # Prefer the largest valid area
            if score > max_score:
                max_score = score
                # Convert back to global coordinates
                best_box = (y_min + y, y_min + y + ch, x, x + cw)

    # Fallback: If auto-detect fails, use smart logic based on image height
    if best_box is None:
        print("! Auto-detect weak. Using calculated position based on image height.")
        if h > 800: # Full Screenshot Fallback
            # Approx location for 1080p
            best_box = (660, 715, 15, 500)
        else:       # Cropped Image Fallback
            best_box = (120, 175, 10, 490)
            
    return best_box

def extract_chart_tabs(input_path, output_dir):
    img = cv2.imread(str(input_path))
    if img is None: return

    print(f"Analyzing {input_path.name}...")

    # --- Step 1: Find the Strip Automatically ---
    y1, y2, x1, x2 = detect_tab_strip_smart(img)
    
    # Refine: Add a tiny padding
    y1 = max(0, y1 - 2)
    y2 = min(img.shape[0], y2 + 2)
    x1 = max(0, x1 - 2)
    x2 = min(img.shape[1], x2 + 2)
    
    print(f"  → Found Tab Strip at: y={y1}-{y2}, x={x1}-{x2}")
    
    # --- Step 2: Divide into 5 Equal Blocks ---
    strip_width = x2 - x1
    tab_width = strip_width / 5.0
    boundaries = [0]
    for i in range(1, 6):
        boundaries.append(int(i * tab_width))
        
    # --- Step 3: Extract and Save ---
    chart_tabs_img = img[y1:y2, x1:x2]
    
    safe_name = input_path.stem.replace('roi0_bottom_half_', '').replace(' ', '_')
    output_filename = f"roi5_chart_tabs_{safe_name}.png"
    output_path = output_dir / output_filename
    cv2.imwrite(str(output_path), chart_tabs_img)
    
    # --- Step 4: Visualize & Detect ---
    visualize_result(img, (y1, y2, x1, x2), boundaries, output_dir, safe_name)
    detect_selected_tab(chart_tabs_img, output_path, boundaries)

def visualize_result(full_img, coords, boundaries, output_dir, name):
    viz = full_img.copy()
    y1, y2, x1, x2 = coords
    
    for i in range(len(boundaries) - 1):
        bx1 = x1 + boundaries[i]
        bx2 = x1 + boundaries[i+1]
        
        # Draw Cyan Box
        cv2.rectangle(viz, (bx1, y1), (bx2, y2), (255, 255, 0), 2)
        # Label
        cv2.putText(viz, f"C{i+1}", (bx1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    viz_path = output_dir / f"viz_blocks_{name}.png"
    cv2.imwrite(str(viz_path), viz)
    print(f"  ✓ Saved visualization: {viz_path}")

def detect_selected_tab(strip_img, base_path, boundaries):
    """Check which of the 5 segments has the 'active' color (Yellow/Orange)."""
    hsv = cv2.cvtColor(strip_img, cv2.COLOR_BGR2HSV)
    
    # Topcon Active Tab Color Range (Yellow/Orange)
    lower_active = np.array([10, 80, 80])
    upper_active = np.array([40, 255, 255])
    
    max_score = 0
    selected_idx = -1
    
    for i in range(5):
        bx1, bx2 = boundaries[i], boundaries[i+1]
        roi = hsv[:, bx1:bx2]
        
        mask = cv2.inRange(roi, lower_active, upper_active)
        score = np.sum(mask > 0) / roi.size
        
        if score > max_score:
            max_score = score
            selected_idx = i
            
    # Output Logic
    txt_path = str(base_path).replace('.png', '_selected.txt')
    with open(txt_path, 'w') as f:
        if max_score > 0.05: 
            f.write(f"selected_index: {selected_idx}\n")
            f.write(f"selected_chart: Chart{selected_idx+1}\n")
            f.write(f"confidence: {max_score:.3f}")
            print(f"  → Selected: Chart{selected_idx+1} (Conf: {max_score:.2f})")
        else:
            f.write("selected_index: -1\nstatus: None Selected")
            print("  → No active tab detected.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Point this to where your images are. 
    # If you put full screenshots in 'ROI_0' and cropped in 'ROI_5', check both?
    # For now, let's look in ROI_5 as that is your workflow.
    input_dir = Path("ROI_5")
    output_dir = Path("ROI_5")      
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"Error: Folder '{input_dir}' not found.")
        exit()

    # Get Images
    inputs = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    # Filter out output files (viz_, roi5_)
    inputs = [x for x in inputs if not (x.name.startswith("viz_") or x.name.startswith("roi5_"))]
    
    print(f"Found {len(inputs)} input images in '{input_dir}'\n")
    
    for img_file in inputs:
        extract_chart_tabs(img_file, output_dir)
        print("-" * 30)