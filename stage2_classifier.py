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
    """
    if roi_img is None or roi_img.size == 0:
        return 'unknown'
        
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    
    # Red: lowered Saturation and Value for robustness
    red_lower1 = np.array([0, 80, 80])
    red_upper1 = np.array([15, 255, 255])
    red_lower2 = np.array([155, 80, 80])
    red_upper2 = np.array([180, 255, 255])
    
    # Green: lowered Saturation and Value
    green_lower = np.array([40, 80, 80])
    green_upper = np.array([90, 255, 255])
    
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, red_lower1, red_upper1), 
                              cv2.inRange(hsv, red_lower2, red_upper2))
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    
    min_pixels = 40 
    
    if red_pixels > green_pixels and red_pixels > min_pixels:
        return 'red'
    elif green_pixels > red_pixels and green_pixels > min_pixels:
        return 'green'
    else:
        return 'unknown'

def detect_line_presence(roi_img, cyl_axis=None):
    """
    Distinguish Power Refine vs Axis Refine based on handle and dot geometry.
    
    Logic (Internal Rotation-Invariant):
    Axis Refine: Red/White dots are at +/- 45 degrees to the handle.
    Power Refine: Red/White dots are aligned with the handle (0 or 90 degrees).
    """
    if roi_img is None or roi_img.size == 0:
        return False
        
    h, w = roi_img.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # 1. Detect dots
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 50, 40]), np.array([20, 255, 255])),
                              cv2.inRange(hsv, np.array([150, 50, 40]), np.array([180, 255, 255])))
    mask_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 80, 255]))
    mask_dots = cv2.bitwise_or(mask_red, mask_white)
    
    # Limit search to 85% of ROI to avoid circumference noise
    draw_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(draw_mask, (center_x, center_y), int(min(center_x, center_y) * 0.85), 255, -1)
    mask_dots = cv2.bitwise_and(mask_dots, draw_mask)
    
    contours, _ = cv2.findContours(mask_dots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots_angles = []
    for cnt in contours:
        if 5 < cv2.contourArea(cnt) < 500:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                dots_angles.append((np.arctan2(int(M["m01"]/M["m00"]) - center_y, 
                                               int(M["m10"]/M["m00"]) - center_x) * 180 / np.pi) % 180)

    if not dots_angles:
        return False # No data to decide

    # 2. Detect the handle (longest line)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=25, maxLineGap=10)
    
    best_handle_angle = None
    if lines is not None:
        max_len = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > max_len:
                max_len = length
                best_handle_angle = (np.arctan2(y2-y1, x2-x1) * 180 / np.pi) % 180

    # 3. Decision Logic
    # We determine if dots are offset (Axis) or aligned (Power) with the handle.
    
    # If no handle detected, fallback to the mirrored OCR axis as our 'virtual handle'
    ref_angle = None
    if best_handle_angle is not None:
        ref_angle = best_handle_angle
    elif cyl_axis is not None:
        try:
            # Screen Y down means OCR angle is mirrored (180 - angle)
            ref_angle = (180 - float(cyl_axis)) % 180
        except: pass
        
    if ref_angle is None:
        ref_angle = 0.0 # Final fallback

    power_votes = 0
    axis_votes = 0
    for dot_angle in dots_angles:
        diff = abs(dot_angle - ref_angle)
        if diff > 90: diff = abs(180 - diff)
        
        # Power dots are at 0 or 90 relative to handle
        # Axis dots are at 45 relative to handle
        normalized_to_45 = abs(diff)
        if normalized_to_45 > 45: normalized_to_45 = abs(90 - normalized_to_45)
        
        if normalized_to_45 < 22.5: # closer to 0/90
            power_votes += 1
        else: # closer to 45
            axis_votes += 1
            
    return power_votes >= axis_votes

def detect_line_presence_fallback(roi_img, cyl_axis=None):
    """Fallback: logic unified in detect_line_presence"""
    return detect_line_presence(roi_img, cyl_axis=cyl_axis)

def classify_jcc_pattern(roi_img, cyl_axis=None):
    """
    Classify JCC patterns using robust handle-centric logic.
    """
    color = detect_circumference_color(roi_img)
    
    if color == 'unknown':
        return classify_filled(roi_img)
    
    # Detect pattern type (Axis vs Power)
    is_power = detect_line_presence(roi_img, cyl_axis=cyl_axis)
    pattern_type = 'power' if is_power else 'axis'
    
    return f"{color}_{pattern_type}_refine"

if __name__ == "__main__":
    # Test stub
    pass
