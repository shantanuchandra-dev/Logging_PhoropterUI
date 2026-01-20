import cv2
import numpy as np
import pytesseract
import os
import datetime
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def extract_pd_value(pd_crop_img):
    """
    Extract the PD value from a cropped PD region image (numpy array).
    Returns the extracted PD value as a string (or None if not found).
    """
    if pd_crop_img is None or pd_crop_img.size == 0:
        return None

    # Save the first 20 crops for debugging
    if not hasattr(extract_pd_value, "_debug_count"):
        extract_pd_value._debug_count = 0
    if extract_pd_value._debug_count < 20:
        os.makedirs("ROI_2", exist_ok=True)
        cv2.imwrite(f"ROI_2/pd_crop_debug_{extract_pd_value._debug_count+1:02d}.png", pd_crop_img)
        extract_pd_value._debug_count += 1

    # Preprocess for OCR (similar to logic in extract())
    pd_gray = cv2.cvtColor(pd_crop_img, cv2.COLOR_BGR2GRAY)
    pd_res = cv2.resize(pd_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, pd_bin = cv2.threshold(pd_res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
    pd_text = None
    try:
        pd_text = pytesseract.image_to_string(pd_bin, config=custom_config).strip()
    except Exception as e:
        try:
            import easyocr
            import logging
            logging.getLogger('easyocr').setLevel(logging.ERROR)
            reader = easyocr.Reader(['en'], gpu=False)
            results = reader.readtext(pd_bin, detail=0)
            pd_text = " ".join(results).strip() if results else None
        except Exception as e2:
            pd_text = None
    return pd_text

# Try to find tesseract
tesseract_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Users\chirayu.maru\AppData\Local\Tesseract-OCR\tesseract.exe',
    r'C:\Users\chirayu.maru\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
    '/usr/local/bin/tesseract',
    '/opt/homebrew/bin/tesseract'
]

for path in tesseract_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break


def extract(roi0_img, save_debug=False, output_dir='ROI_2', filename=None):
    """
    Extract PD (Pupillary Distance) value from ROI-0 image array.
    Finds the PD box between two occluder circles.
    Args:
        roi0_img: ROI-0 image (numpy array)
        save_debug: Whether to save debug images
        output_dir: Directory to save debug images
    Returns:
        dict: {
            'roi_id': 'ROI_2',
            'pd_value_bbox': [x, y, w, h],  # PD value box coordinates
            'pd_value': str,  # Extracted PD value
            'confidence': None,  # Not available with pytesseract
            'image_path': 'path/to/debug_image.png' (if save_debug=True)
        }
    """
    if roi0_img is None:
        raise ValueError('Input image is None')
    # Use filename for debug output naming if provided
    if filename:
        input_base = os.path.splitext(os.path.basename(filename))[0]
        prefix = input_base[:4]
    else:
        prefix = 'roi0'
    # Resize to expected resolution for ROI 2
    img = cv2.resize(roi0_img, (929, 823))
    h, w = img.shape[:2]

    # Find Circles (Occluders)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough Circles to find the occluder circles
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                               param1=50, param2=35, minRadius=30, maxRadius=70)

    if circles is None:
        return {
            'roi_id': 'ROI_2',
            'pd_value_bbox': [],
            'pd_value': None,
            'confidence': None,
            'error': 'No circles detected'
        }

    circles = np.uint16(np.around(circles))
    detected_circles = sorted(circles[0, :], key=lambda x: x[0])
    
    if len(detected_circles) < 2:
        return {
            'roi_id': 'ROI_2',
            'pd_value_bbox': [],
            'pd_value': None,
            'confidence': None,
            'error': f'Found {len(detected_circles)} circles, need at least 2'
        }

    # Find the two circles closest to the center vertically
    mid_y = h / 2
    candidates = sorted(detected_circles, key=lambda c: abs(c[1] - mid_y))
    left_right = sorted(candidates[:2], key=lambda c: c[0])
    
    left_circle = left_right[0]
    right_circle = left_right[1]

    # Detect the rectangle between 2 circles
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
    
    # Find the rectangle (PD box) using Hough Lines
    pd_gray = cv2.cvtColor(roi_pd_area, cv2.COLOR_BGR2GRAY)
    pd_edges = cv2.Canny(pd_gray, 50, 150)
    
    # Use HoughLinesP to find box edges
    lines = cv2.HoughLinesP(pd_edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)
    
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            # Check if horizontal or vertical
            if abs(ly1 - ly2) < 5:
                horizontal_lines.append(ly1)
            elif abs(lx1 - lx2) < 5:
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
            aspect = bw / float(bh) if bh > 0 else 0
            if 1.0 < aspect < 2.5 and 50 < bw < 200 and 30 < bh < 120:
                if area > max_area:
                    max_area = area
                    pd_box = (bx, by, bw, bh)

    if pd_box is None:
        return {
            'roi_id': 'ROI_2',
            'pd_value_bbox': [],
            'pd_value': None,
            'confidence': None,
            'error': 'PD box rectangle not found'
        }

    bx, by, bw, bh = pd_box
    roi_pd_box = roi_pd_area[by:by+bh, bx:bx+bw]
    
    # Extract PD value - refine by detecting the exact bounding box of the digits
    # The PD label is usually at the top, value at the bottom.
    
    # Preprocess for finding digits within the box
    roi_pd_box_gray_full = cv2.cvtColor(roi_pd_box, cv2.COLOR_BGR2GRAY)
    _, roi_pd_box_bin_inv = cv2.threshold(roi_pd_box_gray_full, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours within the PD box to isolate digits
    digit_contours, _ = cv2.findContours(roi_pd_box_bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    val_x1, val_y1, val_x2, val_y2 = bw, bh, 0, 0
    found_digits = False
    
    for cnt in digit_contours:
        dx, dy, dw, dh = cv2.boundingRect(cnt)
        # Filter digits: usually in the lower 70% and within reasonable size ranges
        if dy > bh * 0.3 and 2 < dw < bw * 0.5 and 5 < dh < bh:
            val_x1 = min(val_x1, dx)
            val_y1 = min(val_y1, dy)
            val_x2 = max(val_x2, dx + dw)
            val_y2 = max(val_y2, dy + dh)
            found_digits = True
            
    if found_digits:
        # Add a 2px buffer
        buffer = 2
        val_x1 = max(0, val_x1 - buffer)
        val_y1 = max(0, val_y1 - buffer)
        val_x2 = min(bw, val_x2 + buffer)
        val_y2 = min(bh, val_y2 + buffer)
        
        val_bx, val_by, val_bw, val_bh = val_x1, val_y1, val_x2 - val_x1, val_y2 - val_y1
        roi_pd_value_crop = roi_pd_box[val_by:val_by+val_bh, val_bx:val_bx+val_bw]
    else:
        # Fallback to bottom 60% if no digits detected reliably
        val_by = int(bh * 0.4)
        val_bh = bh - val_by
        val_bx = 0
        val_bw = bw
        roi_pd_value_crop = roi_pd_box[val_by:, :]
    
    # Preprocess for OCR
    roi_pd_box_gray = cv2.cvtColor(roi_pd_value_crop, cv2.COLOR_BGR2GRAY)
    roi_pd_box_res = cv2.resize(roi_pd_box_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, roi_pd_box_bin = cv2.threshold(roi_pd_box_res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
    pd_text = None
    
    try:
        pd_text = pytesseract.image_to_string(roi_pd_box_bin, config=custom_config).strip()
    except Exception as e:
        try:
            import easyocr
            # Suppress easyocr loading output
            import logging
            logging.getLogger('easyocr').setLevel(logging.ERROR)
            
            reader = easyocr.Reader(['en'], gpu=False)
            results = reader.readtext(roi_pd_box_bin, detail=0)
            pd_text = " ".join(results).strip() if results else None
        except Exception as e2:
            pd_text = None

    # Calculate absolute coordinates of the refined value box in resized space
    val_bx_abs = search_x1 + bx + val_bx
    val_by_abs = search_y1 + by + val_by
    
    # Scale coordinates back to original image resolution
    orig_h, orig_w = roi0_img.shape[:2]
    scale_x = orig_w / 929.0
    scale_y = orig_h / 823.0
    
    full_bx_val = val_bx_abs * scale_x
    full_by_val = val_by_abs * scale_y
    val_bw_scaled = val_bw * scale_x
    val_bh_scaled = val_bh * scale_y
    
    result = {
        'roi_id': 'ROI_2',
        'pd_value_bbox': [int(full_bx_val), int(full_by_val), int(val_bw_scaled), int(val_bh_scaled)],
        'pd_value': pd_text,
        'confidence': None
    }
    
    if save_debug:
        os.makedirs(output_dir, exist_ok=True)
        now = datetime.datetime.now().strftime('%d%m_%H%M%S')
        # Save the PD crop (the value part)
        result_path = os.path.join(output_dir, f'{prefix}_{now}_pd_debug.png')
        cv2.imwrite(result_path, roi_pd_value_crop)
        # Save visualization on the full image (using original image resolution)
        vis_full = roi0_img.copy()
        # Scale circle coordinates for visualization if needed, but here we just show the final bbox
        # Note: left_circle/right_circle are in resized space, let's scale them too if we draw them
        lx_scaled = int(left_circle[0] * scale_x)
        ly_scaled = int(left_circle[1] * scale_y)
        lr_scaled = int(left_circle[2] * ((scale_x + scale_y) / 2)) # Approx scale for radius
        rx_scaled = int(right_circle[0] * scale_x)
        ry_scaled = int(right_circle[1] * scale_y)
        rr_scaled = int(right_circle[2] * ((scale_x + scale_y) / 2))
        
        cv2.circle(vis_full, (lx_scaled, ly_scaled), lr_scaled, (255, 0, 0), 2)
        cv2.circle(vis_full, (rx_scaled, ry_scaled), rr_scaled, (255, 0, 0), 2)
        cv2.rectangle(vis_full, (int(full_bx_val), int(full_by_val)), (int(full_bx_val + val_bw_scaled), int(full_by_val + val_bh_scaled)), (0, 255, 0), 2)
        vis_path = os.path.join(output_dir, f'{prefix}_{now}_pd_vis.png')
        cv2.imwrite(vis_path, vis_full)
        result['image_path'] = result_path
    
    return result


if __name__ == "__main__":
    # Fallback to loading from ROI_0 directory
    roi0_dir = 'ROI_0'
    roi0_files = [f for f in os.listdir(roi0_dir) if f.endswith('.png') and 'box' not in f]
    if not roi0_files:
        print('No ROI-0 images found in ROI_0 directory.')
        exit(1)
    roi0_files.sort()
    roi0_path = os.path.join(roi0_dir, roi0_files[-1])
    img = cv2.imread(roi0_path)
    if img is None:
        print(f'Could not load {roi0_path}')
        exit(1)
    # Attach filename for debug naming
    img.filename = roi0_path
    # Call the extract function with debug saving enabled
    result = extract(img, save_debug=True)
    print(f'PD extraction result: {result}')
