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
    
    # Green: Narrowed Hue and increased S/V to avoid leakage from blue
    green_lower = np.array([45, 100, 100])
    green_upper = np.array([85, 255, 255])
    
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, red_lower1, red_upper1), 
                              cv2.inRange(hsv, red_lower2, red_upper2))
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    
    total_pixels = roi_img.shape[0] * roi_img.shape[1]
    # Dynamic threshold: 2% of the ROI area (approx 120 pixels for a 60x60 ROI)
    min_pixels = int(total_pixels * 0.02)
    
    if red_pixels > green_pixels and red_pixels > min_pixels:
        return 'red'
    elif green_pixels > red_pixels and green_pixels > min_pixels:
        return 'green'
    else:
        return 'unknown'

def detect_line_presence(roi_img, cyl_axis=None):
    """
    Distinguish Power Refine vs Axis Refine based on handle and dot geometry.
    Returns: bool (True for Power, False for Axis) or None if no structure found.
    """
    if roi_img is None or roi_img.size == 0:
        return None
        
    h, w = roi_img.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # 1. Detect dots using contours (Primary)
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 50, 40]), np.array([20, 255, 255])),
                              cv2.inRange(hsv, np.array([150, 50, 40]), np.array([180, 255, 255])))
    mask_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 80, 255]))
    mask_dots = cv2.bitwise_or(mask_red, mask_white)
    
    # NEW: Tighten search to 60% of ROI to strictly avoid circumference noise
    draw_mask = np.zeros((h, w), dtype=np.uint8)
    search_radius = int(min(center_x, center_y) * 0.60)
    cv2.circle(draw_mask, (center_x, center_y), search_radius, 255, -1)
    mask_dots_filtered = cv2.bitwise_and(mask_dots, draw_mask)
    
    contours, _ = cv2.findContours(mask_dots_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots_angles = []
    for cnt in contours:
        if 5 < cv2.contourArea(cnt) < 500:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                dots_angles.append((np.arctan2(int(M["m01"]/M["m00"]) - center_y, 
                                               int(M["m10"]/M["m00"]) - center_x) * 180 / np.pi) % 180)

    # 2. Detect the handle (Structural Pivot)
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

    if not dots_angles and best_handle_angle is None:
        return None  # No clinical structure found

    # 3. Decision Logic
    ref_angle = None
    if best_handle_angle is not None:
        ref_angle = best_handle_angle
    elif cyl_axis is not None:
        try:
            ref_angle = (180 - float(cyl_axis)) % 180
        except: pass
        
    if ref_angle is None:
        ref_angle = 0.0

    # TIE-BREAKER: Structural Profiling (Sampling along axes)
    # If no dots or tied votes, we sample pixel intensities along Power and Axis radial lines.
    def get_axis_score(angles_deg):
        score = 0
        for angle in angles_deg:
            # Sample at 3 points along the radial line (30%, 45%, 60% of radius)
            for r_fact in [0.3, 0.45, 0.6]:
                rad = angle * np.pi / 180.0
                px = int(center_x + search_radius * r_fact * np.cos(rad))
                py = int(center_y + search_radius * r_fact * np.sin(rad))
                if 0 <= py < h and 0 <= px < w:
                    # Check if pixel is "dot-like" (in our color masks)
                    if mask_dots[py, px] > 0:
                        score += 1
        return score

    # Power axes: 0, 90, 180, 270 relative to handle (FULL CIRCLE for cos/sin)
    power_axes = [ref_angle, ref_angle + 90, ref_angle + 180, ref_angle + 270]
    # Axis axes: 45, 135, 225, 315 relative to handle
    axis_axes = [ref_angle + 45, ref_angle + 135, ref_angle + 225, ref_angle + 315]

    p_score = get_axis_score(power_axes)
    a_score = get_axis_score(axis_axes)

    if not dots_angles:
        # If no dots detected via contours, use profiling score
        if p_score > a_score: return True
        if a_score > p_score: return False
        return True # Default to Power on absolute tie

    power_votes = 0
    axis_votes = 0
    for dot_angle in dots_angles:
        diff = abs(dot_angle - ref_angle)
        if diff > 90: diff = abs(180 - diff)
        
        normalized_to_45 = abs(diff)
        if normalized_to_45 > 45: normalized_to_45 = abs(90 - normalized_to_45)
        
        if normalized_to_45 < 22.5: # closer to 0/90
            power_votes += 1
        else: # closer to 45
            axis_votes += 1
            
    if power_votes == axis_votes:
        return p_score >= a_score
        
    return power_votes > axis_votes

def classify_jcc_pattern(roi_img, cyl_axis=None):
    """
    Classify JCC patterns using robust handle-centric logic.
    """
    color = detect_circumference_color(roi_img)
    
    if color == 'unknown':
        return classify_filled(roi_img)
    
    # STRUCTURAL VALIDATION: Detect pattern type (Axis vs Power)
    # This also confirms if ANY handle/dots exist.
    is_power = detect_line_presence(roi_img, cyl_axis=cyl_axis)
    
    if is_power is None:
        # Color was detected but no dots/handle structure exists. 
        # Fallback to filled state to avoid false refined identifications.
        return classify_filled(roi_img)
        
    pattern_type = 'power' if is_power else 'axis'
    return f"{color}_{pattern_type}_refine"

if __name__ == "__main__":
    # Test stub
    pass
