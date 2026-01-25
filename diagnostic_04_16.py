import cv2
import os
import json
import numpy as np
import extract_roi0
import extract_roi3_4_jcc_SC as extract_roi3_4
from stage2_classifier import detect_circumference_color, detect_line_presence, classify_jcc_pattern

def debug_detect_line_presence_verbose(roi_img, cyl_axis=None):
    print(f"  --- Verbose Line Presence Detection (cyl_axis={cyl_axis}) ---")
    if roi_img is None or roi_img.size == 0:
        return None
        
    h, w = roi_img.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # 1. Detect dots
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 50, 40]), np.array([20, 255, 255])),
                              cv2.inRange(hsv, np.array([150, 50, 40]), np.array([180, 255, 255])))
    mask_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 80, 255]))
    mask_dots = cv2.bitwise_or(mask_red, mask_white)
    
    draw_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(draw_mask, (center_x, center_y), int(min(center_x, center_y) * 0.85), 255, -1)
    mask_dots = cv2.bitwise_and(mask_dots, draw_mask)
    
    contours, _ = cv2.findContours(mask_dots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots_angles = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if 5 < area < 500:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                angle = (np.arctan2(int(M["m01"]/M["m00"]) - center_y, 
                                   int(M["m10"]/M["m00"]) - center_x) * 180 / np.pi) % 180
                dots_angles.append(angle)
                print(f"    Dot {i}: angle={angle:.1f}, area={area:.1f}")

    # 2. Detect the handle
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
        print(f"    Handle detected at angle: {best_handle_angle:.1f}")

    if not dots_angles and best_handle_angle is None:
        print("    No dots or handle found.")
        return None

    # 3. Decision Logic
    ref_angle = None
    if best_handle_angle is not None:
        ref_angle = best_handle_angle
        print(f"    Using handle as ref: {ref_angle:.1f}")
    elif cyl_axis is not None:
        try:
            ref_angle = (180 - float(cyl_axis)) % 180
            print(f"    Using cyl_axis as ref: {ref_angle:.1f}")
        except: pass
        
    if ref_angle is None:
        ref_angle = 0.0
        print("    Using default ref: 0.0")

    power_votes = 0
    axis_votes = 0
    for dot_angle in dots_angles:
        diff = abs(dot_angle - ref_angle)
        if diff > 90: diff = abs(180 - diff)
        
        normalized_to_45 = abs(diff)
        if normalized_to_45 > 45: normalized_to_45 = abs(90 - normalized_to_45)
        
        print(f"    Dot vs Ref: angle={dot_angle:.1f}, diff={diff:.1f}, norm_45={normalized_to_45:.1f}")
        
        if normalized_to_45 < 22.5: # closer to 0/90
            power_votes += 1
            print("      Vote: POWER")
        else: # closer to 45
            axis_votes += 1
            print("      Vote: AXIS")
            
    print(f"    Final Votes -> Power: {power_votes}, Axis: {axis_votes}")
    return power_votes >= axis_votes

def debug_frame(frame_path, r_ax, l_ax):
    print(f"\n============================================================")
    print(f"DEBUGGING FRAME: {frame_path} (R_AX={r_ax}, L_AX={l_ax})")
    print(f"============================================================")
    frame = cv2.imread(frame_path)
    if frame is None:
        return

    r0 = extract_roi0.extract_roi0(frame)['roi0']
    
    res = extract_roi3_4.extract(r0, save_debug=True, filename=f"debug_{frame_path}", right_axis=r_ax, left_axis=l_ax)
    
    print(f"\nFinal Phoropter State: {res.get('phoropter_state')}")
    
    for bb in res.get('bboxes', []):
        label = bb['label']
        box = bb['box']
        print(f"\nOCULDER: {label}")
        
        x, y, w, h = box
        roi_img = r0[y:y+h, x:x+w]
        
        axis_for_eye = r_ax if label == 'right_occluder' else l_ax
        debug_detect_line_presence_verbose(roi_img, cyl_axis=axis_for_eye)

if __name__ == "__main__":
    # Test Frame 04:16 (Supposed to be Power)
    debug_frame("frame_04_16.png", r_ax=180, l_ax=180)
    
    # Test Frame 04:04 (Supposed to be Axis?)
    debug_frame("frame_04_04.png", r_ax=170, l_ax=180)
