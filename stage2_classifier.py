"""
Stage 2 Rule-Based Classifiers
Detailed classification after Stage 1 pattern type detection
"""

import cv2
import numpy as np

def classify_filled(roi_img):
    """
    Classify filled circles as grey_filled or blue_filled.
    Uses color analysis in HSV space for better robustness.
    
    Args:
        roi_img: OpenCV image (BGR) of the occluder
    
    Returns:
        str: 'grey_filled' or 'blue_filled'
    """
    if roi_img is None or roi_img.size == 0:
        return 'grey_filled'
        
    # Convert to HSV
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    
    # Define blue range in HSV
    # Blue is typically H: 90-130, S: 50-255, V: 50-255
    blue_lower = np.array([90, 40, 40])
    blue_upper = np.array([130, 255, 255])
    
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blue_pixels = cv2.countNonZero(blue_mask)
    total_pixels = roi_img.shape[0] * roi_img.shape[1]
    
    # If more than 30% of pixels are blue, consider it a blue occluder
    if blue_pixels / total_pixels > 0.3:
        return 'blue_filled'
    else:
        return 'grey_filled'

def detect_circumference_color(roi_img):
    """
    Detect if the circumference is red or green.
    Uses HSV color space with calibrated thresholds.
    
    Args:
        roi_img: OpenCV image (BGR) of the occluder
    
    Returns:
        str: 'red', 'green', or 'unknown'
    """
    if roi_img is None or roi_img.size == 0:
        return 'unknown'
        
    # Convert to HSV
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    # Red: lowered Saturation and Value for robustness
    red_lower1 = np.array([0, 80, 80])
    red_upper1 = np.array([15, 255, 255])
    red_lower2 = np.array([155, 80, 80])
    red_upper2 = np.array([180, 255, 255])
    
    # Green: lowered Saturation and Value
    green_lower = np.array([40, 80, 80])
    green_upper = np.array([90, 255, 255])
    
    # Create masks
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, red_lower1, red_upper1), 
                              cv2.inRange(hsv, red_lower2, red_upper2))
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # Count pixels
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    
    # min_pixels threshold to avoid noise
    min_pixels = 40 
    
    if red_pixels > green_pixels and red_pixels > min_pixels:
        return 'red'
    elif green_pixels > red_pixels and green_pixels > min_pixels:
        return 'green'
    else:
        return 'unknown'

def detect_line_presence(roi_img):
    """
    Distinguish Power Refine vs Axis Refine using dot geometry.
    
    Power Refine: Dots are axis-aligned (left, right, top, bottom).
    Axis Refine: Dots are diagonal (relative to horizontal/vertical).
    
    Returns:
        bool: True if it's Power Refine, False if it's Axis Refine
    """
    if roi_img is None or roi_img.size == 0:
        return False
        
    h, w = roi_img.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # 1. Improved Dot Detection (Red and White)
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    
    # Red dots
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([20, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([150, 50, 40]), np.array([180, 255, 255]))
    red_dots = cv2.bitwise_or(mask_red1, mask_red2)
    
    # White dots
    white_dots = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 80, 255]))
    
    all_dots_mask = cv2.bitwise_or(red_dots, white_dots)
    
    # 1b. Exclude the outer ring to avoid merging dots with circumference
    # Assuming dots are within 70% of the radius from center
    draw_mask = np.zeros((h, w), dtype=np.uint8)
    inner_radius = int(min(center_x, center_y) * 0.75)
    cv2.circle(draw_mask, (center_x, center_y), inner_radius, 255, -1)
    all_dots_mask = cv2.bitwise_and(all_dots_mask, draw_mask)
    
    kernel = np.ones((3,3), np.uint8)
    all_dots_mask = cv2.morphologyEx(all_dots_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(all_dots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 4 < area < 400:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                if 5 < cx < w-5 and 5 < cy < h-5:
                    dots.append((cx, cy))
                    
    # 2. Geometric Classification
    if len(dots) >= 2:
        power_votes = 0
        axis_votes = 0
        for dx, dy in dots:
            vx, vy = dx - center_x, dy - center_y
            if abs(vx) < 3 and abs(vy) < 3: continue
            
            angle = abs(np.arctan2(vy, vx) * 180 / np.pi)
            # Normalize to [0, 45]
            if angle > 90: angle = 180 - angle
            if angle > 45: angle = 90 - angle # angle is distance to closest axis (0 or 90)
            
            if angle < 22.5: # Closer to axis than diagonal
                power_votes += 1
            else:
                axis_votes += 1
        return power_votes >= axis_votes

    # Fallback to simple line detection if dots failed
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=10)
    return lines is not None



def classify_jcc_pattern(roi_img):
    """
    Classify JCC patterns into one of 4 classes.
    Falls back to filled if no valid red/green found.
    """
    # Detect color (red vs green)
    color = detect_circumference_color(roi_img)
    
    if color == 'unknown':
        # Default to blue_filled if it's not a clear JCC pattern
        return classify_filled(roi_img)
    
    # Detect pattern type (axis vs power)
    has_line = detect_line_presence(roi_img)
    pattern_type = 'power' if has_line else 'axis'
    
    return f"{color}_{pattern_type}_refine"


# Test functions
if __name__ == "__main__":
    import os
    
    print("=" * 70)
    print("Stage 2 Rule-Based Classifier Tests")
    print("=" * 70)
    
    # Test filled classification
    print("\n### Testing Filled Classification ###")
    
    blue_sample = cv2.imread('ROI_3/cWLM_2201_135636_roi3.png')
    if blue_sample is not None:
        result = classify_filled(blue_sample)
        print(f"Blue sample: {result} (expected: blue_filled)")
    
    grey_sample = cv2.imread('jcc_occluder_dataset/grey_filled/synth_grey_0.png')
    if grey_sample is not None:
        result = classify_filled(grey_sample)
        print(f"Grey sample: {result} (expected: grey_filled)")
    
    # Test JCC pattern classification
    print("\n### Testing JCC Pattern Classification ###")
    
    test_cases = [
        ('jcc_occluder_dataset/red_axis_refine/Screenshot 2026-01-21 at 13.31.40.png', 'red_axis_refine'),
        ('jcc_occluder_dataset/green_axis_refine/Screenshot 2026-01-21 at 13.31.54.png 14-16-13-503.png', 'green_axis_refine'),
        ('jcc_occluder_dataset/red_power_refine/Screenshot 2026-01-21 at 13.32.35.png', 'red_power_refine'),
        ('jcc_occluder_dataset/green_power_refine/Screenshot 2026-01-21 at 13.33.04.png', 'green_power_refine'),
    ]
    
    for path, expected in test_cases:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                result = classify_jcc_pattern(img)
                status = "✅" if result == expected else "❌"
                print(f"{status} {os.path.basename(path)}: {result} (expected: {expected})")
    
    print("\n" + "=" * 70)
