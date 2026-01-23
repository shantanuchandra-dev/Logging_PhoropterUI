
import cv2
import numpy as np
import os

def debug_detect_line_presence(roi_img, filename_prefix):
    h, w = roi_img.shape[:2]
    debug_vis = roi_img.copy()
    
    # 1. Detect lines
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    # Strict parameters matching stage2_classifier.py
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imwrite(f"{filename_prefix}_edges.png", edges)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                            minLineLength=20, maxLineGap=10)
    
    # Draw ALL lines in faint gray
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_vis, (x1, y1), (x2, y2), (200, 200, 200), 1)
    
    # 2. Find longest line
    longest_line = None
    max_length = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > max_length:
                max_length = length
                longest_line = (x1, y1, x2, y2)
                
    if longest_line:
        lx1, ly1, lx2, ly2 = longest_line
        cv2.line(debug_vis, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2) # Green detected line

    # 3. Detect dots
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    # Red dots
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([20, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([150, 50, 40]), np.array([180, 255, 255]))
    red_dots = cv2.bitwise_or(mask_red1, mask_red2)
    # White dots
    white_dots = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 80, 255]))
    all_dots_mask = cv2.bitwise_or(red_dots, white_dots)
    
    kernel = np.ones((3,3), np.uint8)
    all_dots_mask = cv2.morphologyEx(all_dots_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{filename_prefix}_dots_mask.png", all_dots_mask)

    contours, _ = cv2.findContours(all_dots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Geometric filtering
    img_center_x, img_center_y = w // 2, h // 2
    max_dist_from_center = (w / 2) * 0.85 
    
    cv2.circle(debug_vis, (img_center_x, img_center_y), int(max_dist_from_center), (0, 255, 255), 1) # Yellow filter circle

    dot_circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 500:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy, radius = int(cx), int(cy), int(radius)
            
            dist_to_center = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
            
            if dist_to_center < max_dist_from_center:
                if radius > 3:
                     dot_circles.append((cx, cy, radius))
                     cv2.circle(debug_vis, (cx, cy), radius, (255, 0, 0), 2) # Blue valid dot
            else:
                cv2.circle(debug_vis, (cx, cy), radius, (0, 165, 255), 2) # Orange invalid dot

    # 4. Check all lines with smart scoring
    best_line = None
    max_score = -1
    best_touching_count = 0
    
    if lines is not None:
        print(f"checking {len(lines)} lines...")
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Count touches
            current_touches = 0
            for cx, cy, r in dot_circles:
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([cx, cy])
                
                if np.array_equal(p1, p2):
                    dist = np.linalg.norm(p3 - p1)
                else:
                    dist = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
                
                if dist <= (r + 2):
                    current_touches += 1
            
            score = (current_touches * 1000) + length
            print(f"Line {i}: len={length:.1f}, touches={current_touches}, score={score:.1f}")
            
            if score > max_score:
                max_score = score
                best_line = (x1, y1, x2, y2)
                best_touching_count = current_touches

    # Draw best line
    if best_line:
        lx1, ly1, lx2, ly2 = best_line
        cv2.line(debug_vis, (lx1, ly1), (lx2, ly2), (0, 255, 0), 3) # THICK green line for best choice
        
        # Visualize touches for best line
        for cx, cy, r in dot_circles:
            p1 = np.array([lx1, ly1])
            p2 = np.array([lx2, ly2])
            p3 = np.array([cx, cy])
            if np.array_equal(p1, p2): dist = np.linalg.norm(p3 - p1)
            else: dist = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
            
            is_touching = dist <= (r + 2)
            color = (0, 0, 255) if is_touching else (255, 0, 0)
            cv2.circle(debug_vis, (cx, cy), r, color, 2)
            cv2.putText(debug_vis, f"{dist:.1f}", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    is_power = best_touching_count >= 2
    status = "POWER" if is_power else "AXIS"
    cv2.putText(debug_vis, f"Result: {status} (BestTouch:{best_touching_count})", 
                (5, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imwrite(f"{filename_prefix}_final.png", debug_vis)
    print(f"Processed {filename_prefix}: {status}")

if __name__ == "__main__":
    # Define output directory
    output_dir = "debug_jcc_specific"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define files to debug
    base_dir = "jcc_occluder_dataset"
    files = [
        (os.path.join(base_dir, "red_axis_refine/13.31.40_aug_5.png"), "user_debug_case"),
    ]
    
    for path, prefix in files:
        if os.path.exists(path):
            print(f"Debugging {path}...")
            img = cv2.imread(path)
            # Pass output directory to function or handle it here
            # Ideally update function to accept output_path or modify filenames to include dir
            # For minimal change, let's update the function signature in the next step or hack it here?
            # Better to update the function.
            debug_detect_line_presence(img, os.path.join(output_dir, prefix))
        else:
            print(f"File not found: {path} (Check path)")
