
import cv2
import numpy as np
import pytesseract
import os
import datetime

# Try to find tesseract
tesseract_paths = [
    
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Users\chirayu.maru\AppData\Local\Tesseract-OCR\tesseract.exe',
    r'C:\Users\chirayu.maru\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
]

for path in tesseract_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break

def extract_roi2():
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

    h, w = img.shape[:2]

    # 2. Find Circles (Occluders)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough Circles to find the occluder circles
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

    # 3. Detect the rectangle between 2 circles
    x1, y1, r1 = left_circle
    x2, y2, r2 = right_circle
    
    # Search region for PD box
    search_x1 = int(x1 + r1)
    search_x2 = int(x2 - r2)
    search_y1 = int(min(y1, y2) - 50)
    search_y2 = int(max(y1, y2) + 50)
    
    search_x1 = max(0, search_x1)
    search_x2 = min(w, search_x2)
    search_y1 = max(0, search_y1)
    search_y2 = min(h, search_y2)
    
    roi_pd_area = img[search_y1:search_y2, search_x1:search_x2]
    
    # 4. Find the rectangle (PD box) using Hough Lines
    pd_gray = cv2.cvtColor(roi_pd_area, cv2.COLOR_BGR2GRAY)
    pd_edges = cv2.Canny(pd_gray, 50, 150)
    
    # Use HoughLinesP to find box edges
    lines = cv2.HoughLinesP(pd_edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)
    
    vis_hough = roi_pd_area.copy()
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            # Check if horizontal or vertical
            if abs(ly1 - ly2) < 5:
                cv2.line(vis_hough, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
                horizontal_lines.append(ly1)
            elif abs(lx1 - lx2) < 5:
                cv2.line(vis_hough, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
                vertical_lines.append(lx1)

    # Use horizontal and vertical lines to define the box
    if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
        y_top = min(horizontal_lines)
        y_bottom = max(horizontal_lines)
        x_left = min(vertical_lines)
        x_right = max(vertical_lines)
        pd_box = (x_left, y_top, x_right - x_left, y_bottom - y_top)
    else:
        # Fallback to contour detection
        pd_thresh = cv2.adaptiveThreshold(pd_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(pd_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pd_box = None
        max_area = 0
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            aspect = bw / float(bh)
            if 1.0 < aspect < 2.5 and 50 < bw < 200 and 30 < bh < 120:
                if area > max_area:
                    max_area = area
                    pd_box = (bx, by, bw, bh)

    if pd_box is None:
        print("PD box rectangle not found.")
        return

    bx, by, bw, bh = pd_box
    roi_pd_box = roi_pd_area[by:by+bh, bx:bx+bw]
    
    # 5. Extract PD value - refine by taking only the bottom part of the box
    # The PD label is usually at the top, value at the bottom.
    # We can split the box vertically.
    roi_pd_value_crop = roi_pd_box[int(bh*0.4):, :] # Take bottom 60%
    
    # Preprocess for OCR
    roi_pd_box_gray = cv2.cvtColor(roi_pd_value_crop, cv2.COLOR_BGR2GRAY)
    roi_pd_box_res = cv2.resize(roi_pd_box_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, roi_pd_box_bin = cv2.threshold(roi_pd_box_res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
    try:
        pd_text = pytesseract.image_to_string(roi_pd_box_bin, config=custom_config)
    except Exception as e:
        pd_text = f"OCR Error: {e}"

    # 6. Save results
    output_dir = 'ROI_2'
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the PD crop (the value part)
    result_path = os.path.join(output_dir, f'roi2_PD_{now}.png')
    cv2.imwrite(result_path, roi_pd_value_crop)
    
    # Save Hough visualization
    hough_path = os.path.join(output_dir, f'roi2_hough_{now}.png')
    cv2.imwrite(hough_path, vis_hough)
    
    # Save visualization on the full image
    vis_full = img.copy()
    cv2.circle(vis_full, (left_circle[0], left_circle[1]), left_circle[2], (255, 0, 0), 2)
    cv2.circle(vis_full, (right_circle[0], right_circle[1]), right_circle[2], (255, 0, 0), 2)
    full_bx = search_x1 + bx
    full_by = search_y1 + by
    cv2.rectangle(vis_full, (full_bx, full_by), (full_bx + bw, full_by + bh), (0, 255, 0), 2)
    
    vis_path = os.path.join(output_dir, f'roi2_PD_vis_{now}.png')
    cv2.imwrite(vis_path, vis_full)
    
    print(f"Results saved to {output_dir}")
    print(f"Detected PD: {pd_text.strip()}")

if __name__ == "__main__":
    extract_roi2()
