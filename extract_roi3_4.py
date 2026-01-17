
import cv2
import numpy as np
import os
import datetime

def extract_roi3_4():
    # 1. Path to the latest ROI-0 image
    roi0_dir = 'ROI_0'
    roi0_files = [f for f in os.listdir(roi0_dir) if f.startswith('roi0_') and f.endswith('.png') and 'box' not in f]
    if not roi0_files:
        print('No ROI-0 images found in ROI_0 directory.')
        return
    roi0_files.sort()
    roi0_path = os.path.join(roi0_dir, roi0_files[-1])

    img = cv2.imread(roi0_path)
    if img is None:
        print(f'Could not load {roi0_path}')
        return

    # Resize to the required resolution for this logic
    img = cv2.resize(img, (929, 823))
    h, w = img.shape[:2]

    # 2. Find Circles (Occluders)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough Circles - using same params as ROI-2 for consistency
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                               param1=50, param2=35, minRadius=30, maxRadius=70)

    if circles is None:
        print("No circles detected.")
        return

    circles = np.uint16(np.around(circles))
    detected_circles = sorted(circles[0, :], key=lambda x: x[0])
    
    if len(detected_circles) < 2:
        print(f"Found {len(detected_circles)} circles, need at least 2.")
        return

    # Find the two circles closest to the center vertically
    mid_y = h / 2
    candidates = sorted(detected_circles, key=lambda c: abs(c[1] - mid_y))
    left_right = sorted(candidates[:2], key=lambda c: c[0])
    
    left_circle = left_right[0]
    right_circle = left_right[1]

    print(f"Detected circles at: Left {left_circle[:2]}, Right {right_circle[:2]}")

    # 3. Crop and Classify
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Left Occluder (ROI-3)
    lx, ly, lr = left_circle
    roi3 = img[ly-lr:ly+lr, lx-lr:lx+lr]
    roi3_dir = 'ROI_3'
    os.makedirs(roi3_dir, exist_ok=True)
    roi3_path = os.path.join(roi3_dir, f'roi3_{now}.png')
    cv2.imwrite(roi3_path, roi3)

    # Right Occluder (ROI-4)
    rx, ry, rr = right_circle
    roi4 = img[ry-rr:ry+rr, rx-rr:rx+rr]
    roi4_dir = 'ROI_4'
    os.makedirs(roi4_dir, exist_ok=True)
    roi4_path = os.path.join(roi4_dir, f'roi4_{now}.png')
    cv2.imwrite(roi4_path, roi4)

    # Classification logic (Heuristic: check blue channel dominance)
    def classify_state(circle_img):
        # Average color
        avg_color = cv2.mean(circle_img)[:3]
        # In BGR, blue is index 0
        if avg_color[0] > avg_color[1] + 20 and avg_color[0] > avg_color[2] + 20:
            return "filled/active (blue)"
        else:
            return "unfilled/inactive (gray)"

    state3 = classify_state(roi3)
    state4 = classify_state(roi4)

    print(f"ROI-3 (Left) State: {state3}")
    print(f"ROI-4 (Right) State: {state4}")

if __name__ == "__main__":
    extract_roi3_4()
