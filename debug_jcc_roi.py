
import cv2
import numpy as np
import os

def debug_jcc_roi(image_path):
    roi_img = cv2.imread(image_path)
    if roi_img is None:
        print(f"Error: Could not read image {image_path}")
        return

    h, w = roi_img.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # 1. Color Masking (HSV)
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    
    # Red dots
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([20, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([150, 50, 40]), np.array([180, 255, 255]))
    red_dots_mask = cv2.bitwise_or(mask_red1, mask_red2)
    
    # White dots
    white_dots_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 80, 255]))
    
    all_dots_mask = cv2.bitwise_or(red_dots_mask, white_dots_mask)
    kernel = np.ones((3,3), np.uint8)
    clean_mask = cv2.morphologyEx(all_dots_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    debug_img = roi_img.copy()
    cv2.circle(debug_img, (center_x, center_y), 3, (255, 255, 0), -1) # Center in Cyan
    
    print(f"\nAnalyzing image: {image_path}")
    print(f"Center: ({center_x}, {center_y})")
    
    power_votes = 0
    axis_votes = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(f"Contour area: {area}")
        if 4 < area < 400:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                if 5 < cx < w-5 and 5 < cy < h-5:
                    dots.append((cx, cy))
                    vx, vy = cx - center_x, cy - center_y
                    
                    if abs(vx) < 3 and abs(vy) < 3: 
                        print(f"Dot at ({cx}, {cy}) is too close to center, skipping voting.")
                        cv2.circle(debug_img, (cx, cy), 3, (255, 0, 255), -1) # Skip in Magenta
                        continue
                    
                    angle = abs(np.arctan2(vy, vx) * 180 / np.pi)
                    norm_angle = angle
                    if norm_angle > 90: norm_angle = 180 - norm_angle
                    dist_to_axis = norm_angle
                    if dist_to_axis > 45: dist_to_axis = 90 - dist_to_axis
                    
                    is_power = dist_to_axis < 22.5
                    if is_power:
                        power_votes += 1
                        color = (0, 255, 255) # Yellow for Power
                    else:
                        axis_votes += 1
                        color = (255, 255, 0) # Cyan for Axis
                        
                    print(f"Dot at ({cx}, {cy}): v=({vx}, {vy}), angle={angle:.1f}, dist_to_axis={dist_to_axis:.1f}, vote={'Power' if is_power else 'Axis'}")
                    cv2.circle(debug_img, (cx, cy), 4, color, -1)
                    cv2.line(debug_img, (center_x, center_y), (cx, cy), color, 1)

    print(f"Final Votes: Power={power_votes}, Axis={axis_votes}")
    print(f"Decision: {'Power Refine' if power_votes >= axis_votes else 'Axis Refine'}")
    
    cv2.imwrite("debug_jcc_analysis.png", debug_img)
    cv2.imwrite("debug_jcc_mask.png", clean_mask)
    print("Debug images saved to debug_jcc_analysis.png and debug_jcc_mask.png")

if __name__ == "__main__":
    image_path = "jcc_occluder_dataset/red_axis_refine/Screenshot 2026-01-21 at 13.31.40.png"
    debug_jcc_roi(image_path)
