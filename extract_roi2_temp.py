import cv2
import numpy as np
import pytesseract
import os
import datetime
import easyocr
import re

def extract_roi2_pd(img):
    # Step 1: Use the middle 20% vertically
    now = datetime.datetime.now().strftime('%d%m_%H%M%S')
    output_dir = 'ROI_2'
    os.makedirs(output_dir, exist_ok=True)
    # Get prefix from caller or from global if available
    import inspect
    prefix = 'roi0'
    frame = inspect.currentframe().f_back
    if 'basename' in frame.f_locals:
        prefix = str(frame.f_locals['basename'])[:4]
    elif 'prefix' in frame.f_locals:
        prefix = str(frame.f_locals['prefix'])[:4]
    elif 'roi0_path' in frame.f_globals:
        base = os.path.splitext(os.path.basename(frame.f_globals['roi0_path']))[0]
        prefix = base[:4]
    h_full, w_full = img.shape[:2]
    y1 = int(h_full * 0.40)
    y2 = int(h_full * 0.60)
    middle_crop = img[y1:y2, :]
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_1middle_20pct.png'), middle_crop)

    # Step 2: Preprocessing
    gray = cv2.cvtColor(middle_crop, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_2gray.png'), gray)
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_3enhanced.png'), enhanced)

    # Step 3: OCR to find 'PD' label with multiple attempts
    pd_bbox = None
    ocr_attempts = [
        {'desc': 'psm6', 'config': '--psm 6', 'img': enhanced},
        {'desc': 'psm7', 'config': '--psm 7', 'img': enhanced},
        {'desc': 'psm11', 'config': '--psm 11', 'img': enhanced},
        {'desc': 'psm6_inv', 'config': '--psm 6', 'img': 255 - enhanced},
        {'desc': 'psm7_inv', 'config': '--psm 7', 'img': 255 - enhanced},
        {'desc': 'psm11_inv', 'config': '--psm 11', 'img': 255 - enhanced},
    ]
    for attempt in ocr_attempts:
        ocr_result = pytesseract.image_to_data(attempt['img'], config=attempt['config'], output_type=pytesseract.Output.DICT)
        for i, text in enumerate(ocr_result['text']):
            if text.strip().upper() == 'PD':
                x, y, w, h = ocr_result['left'][i], ocr_result['top'][i], ocr_result['width'][i], ocr_result['height'][i]
                pd_bbox = (x, y, w, h)
                # Visualize PD bbox
                vis = cv2.cvtColor(attempt['img'], cv2.COLOR_GRAY2BGR)
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_4pd_bbox_{attempt["desc"]}.png'), vis)
                break
        if pd_bbox is not None:
            break
    if pd_bbox is None:
        # Save all OCR text for debugging
        with open(os.path.join(output_dir, f'{prefix}_{now}_ocr_texts.txt'), 'w') as f:
            for attempt in ocr_attempts:
                ocr_result = pytesseract.image_to_string(attempt['img'], config=attempt['config'])
                f.write(f"Attempt {attempt['desc']}\n{ocr_result}\n\n")
        raise Exception('PD label not found by OCR.')

    # Step 4: Use contour detection to find value cell below PD label
    x, y, w, h = pd_bbox
    search_margin = 10
    search_y1 = y + h + 2
    search_y2 = min(enhanced.shape[0], search_y1 + int(h * 2.5))
    search_x1 = max(0, x - search_margin)
    search_x2 = min(enhanced.shape[1], x + w + search_margin)
    value_search_region = enhanced[search_y1:search_y2, search_x1:search_x2]
    # Find contours in the search region
    contours, _ = cv2.findContours(value_search_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    value_bbox = None
    max_area = 0
    for cnt in contours:
        x2, y2, w2, h2 = cv2.boundingRect(cnt)
        area = w2 * h2
        # Heuristic: look for the largest box with aspect ratio ~2:1 (typical for PD value cell)
        aspect = w2 / h2 if h2 > 0 else 0
        if 1.2 < aspect < 4.0 and area > max_area:
            max_area = area
            value_bbox = (x2, y2, w2, h2)
    if value_bbox is not None:
        # Adjust bbox to full image coordinates
        value_x1 = search_x1 + value_bbox[0]
        value_y1 = search_y1 + value_bbox[1]
        value_x2 = value_x1 + value_bbox[2]
        value_y2 = value_y1 + value_bbox[3]
    else:
        # Fallback to old method if no contour found
        value_y1 = y + h + 2
        value_y2 = value_y1 + int(h * 1.2)
        value_x1 = x
        value_x2 = x + w
    value_crop = gray[value_y1:value_y2, value_x1:value_x2]
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_5value_crop.png'), value_crop)

    # Mark bounding box for value on the 4th image (PD bbox visualization)
    pd_bbox_img_path = os.path.join(output_dir, f'{prefix}_{now}_4pd_bbox_{attempt["desc"]}.png')
    if os.path.exists(pd_bbox_img_path):
        pd_bbox_img = cv2.imread(pd_bbox_img_path)
        cv2.rectangle(pd_bbox_img, (value_x1, value_y1), (value_x2, value_y2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_5pd_value_bbox.png'), pd_bbox_img)

    # Also mark bounding box on the original ROI_0 image (with respect to ROI_0 coordinates)
    # value_x1, value_y1 are relative to middle_crop, which is offset by y1
    roi0_value_x1 = value_x1
    roi0_value_y1 = y1 + value_y1
    roi0_value_x2 = value_x2
    roi0_value_y2 = y1 + value_y2
    roi0_img_bbox = [roi0_value_x1, roi0_value_y1, roi0_value_x2, roi0_value_y2]
    # Draw on ROI_0 image
    roi0_img_copy = img.copy()
    cv2.rectangle(roi0_img_copy, (roi0_img_bbox[0], roi0_img_bbox[1]), (roi0_img_bbox[2], roi0_img_bbox[3]), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_6value_bbox.png'), roi0_img_copy)

    # Save coordinates in txt file (x, y, w, h) with respect to ROI_0
    bbox_x = roi0_img_bbox[0]
    bbox_y = roi0_img_bbox[1]
    bbox_w = roi0_img_bbox[2] - roi0_img_bbox[0]
    bbox_h = roi0_img_bbox[3] - roi0_img_bbox[1]
    with open(os.path.join(output_dir, f'{prefix}_{now}_6value_bbox.txt'), 'w') as f:
        f.write(f'{bbox_x},{bbox_y},{bbox_w},{bbox_h}\n')

    # Step 5: OCR for value (digits and dot only) using EasyOCR
    # Redo: Read value from 6th image (value_crop) without limitations
    pd_value = None
    
    # Attempt 1: EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)
    # Allow a broader set of characters to ensure we catch the value
    result_easyocr = reader.readtext(value_crop, detail=0) 
    
    if result_easyocr:
        # Join all found text with spaces
        full_text = " ".join(result_easyocr).strip()
        # Regex for XX.X (or XX X) where separators can be dot, comma, or space
        # AND specifically ending with 0 or 5 (steps of 0.5).
        match = re.search(r'(\d{2})([\.\,\s])([05])', full_text)
        if match:
            # We found something look like XX.X or XX X with valid step
            main_val = match.group(1)
            decimal_val = match.group(3)
            pd_value = f"{main_val}.{decimal_val}"

    # Attempt 2: Pytesseract Fallback
    if pd_value is None:
        value_config = '--psm 7'
        value_text = pytesseract.image_to_string(value_crop, config=value_config)
        value_text = value_text.strip().replace('\n', '') 
        
        match = re.search(r'(\d{2})([\.\,\s])([05])', value_text)
        if match:
             main_val = match.group(1)
             decimal_val = match.group(3)
             pd_value = f"{main_val}.{decimal_val}"

    # Step 6: Output
    result = {
        'pd_bbox': [int(value_x1), int(value_y1), int(value_x2-value_x1), int(value_y2-value_y1)],
        'pd_value': pd_value,
        'confidence': None  # pytesseract does not provide confidence for string extraction
    }
    return result

if __name__ == '__main__':
    import sys
    img = None
    if len(sys.argv) > 1:
        # If a path is provided, load the image
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is None:
            print(f'Could not read image: {img_path}')
            sys.exit(1)
    elif 'img' in globals() and isinstance(globals()['img'], np.ndarray):
        # If an image is already loaded in the global scope, use it
        img = globals()['img']
    else:
        # Fallback: process latest ROI-0 image
        roi0_dir = 'ROI_0'
        roi0_files = [f for f in os.listdir(roi0_dir) if f.startswith('crop_') and f.endswith('.png')]
        roi0_files.sort()
        if not roi0_files:
            print('No crop_ images found in ROI_0.')
            sys.exit(1)
        roi0_path = os.path.join(roi0_dir, roi0_files[-1])
        img = cv2.imread(roi0_path)
        if img is None:
            print(f'Could not read image: {roi0_path}')
            sys.exit(1)
    result = extract_roi2_pd(img)
    print(result)
