import cv2
import numpy as np
import os

def debug_circles(roi0_path):
    img = cv2.imread(roi0_path)
    if img is None:
        print(f"Error: Could not load {roi0_path}")
        return

    print(f"Original shape: {img.shape}")
    
    # Resize to expected resolution for ROI 3/4
    target_size = (929, 823)
    img_resized = cv2.resize(img, target_size)
    h, w = img_resized.shape[:2]
    mid_y = h / 2

    # Find Circles
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough Circles
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                               param1=50, param2=35, minRadius=30, maxRadius=70)

    debug_img = img_resized.copy()
    cv2.line(debug_img, (0, int(mid_y)), (w, int(mid_y)), (0, 0, 255), 1)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        detected_circles = circles[0, :]
        
        print(f"Found {len(detected_circles)} circles.")
        for i, (cx, cy, cr) in enumerate(detected_circles):
            dist_to_mid = abs(cy - mid_y)
            print(f"Circle {i}: x={cx}, y={cy}, r={cr}, dist_to_mid={dist_to_mid}")
            
            # Draw all circles in yellow
            cv2.circle(debug_img, (cx, cy), cr, (0, 255, 255), 2)
            cv2.putText(debug_img, f"{i}: d={dist_to_mid:.1f}", (cx - 20, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Highlight the selected two in magenta
        candidates = sorted(detected_circles, key=lambda c: abs(c[1] - mid_y))
        if len(candidates) >= 2:
            left_right = sorted(candidates[:2], key=lambda c: c[0])
            for i, (cx, cy, cr) in enumerate(left_right):
                cv2.circle(debug_img, (cx, cy), cr, (255, 0, 255), 3)
                label = "ROI3" if i == 0 else "ROI4"
                cv2.putText(debug_img, label, (cx - 20, cy + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    output_path = "ROI_0/JCC_Axis_Green_debug_all_circles.png"
    cv2.imwrite(output_path, debug_img)
    print(f"Debug image saved to: {output_path}")

if __name__ == "__main__":
    debug_circles("ROI_0/JCC_Axis_Green.png")
