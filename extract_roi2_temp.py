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
    # Get prefix from the image path if available, else fallback to 'img'
    import inspect
    prefix = 'roi2'
    frame = inspect.currentframe().f_back
    img_path = None
    if 'img_path' in frame.f_locals:
        img_path = frame.f_locals['img_path']
    elif 'roi0_path' in frame.f_globals:
        img_path = frame.f_globals['roi0_path']
    if img_path:
        base = os.path.splitext(os.path.basename(img_path))[0]
        prefix = base[:4]
    elif 'basename' in frame.f_locals:
        prefix = str(frame.f_locals['basename'])[:4]
    elif 'prefix' in frame.f_locals:
        prefix = str(frame.f_locals['prefix'])[:4]
    h_full, w_full = img.shape[:2]
    y1 = int(h_full * 0.40)
    y2 = int(h_full * 0.60)
    middle_crop = img[y1:y2, :]
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_1middle_20pct.png'), middle_crop)

    # Step 2: Preprocessing - try multiple preprocessing methods
    gray = cv2.cvtColor(middle_crop, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_2gray.png'), gray)
    
    # Multiple preprocessing approaches
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_3enhanced.png'), enhanced)
    
    # Additional preprocessing variants
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    enhanced_clahe = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    
    # Simple threshold
    _, thresh_simple = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Step 3: OCR to find 'PD' label with multiple attempts and preprocessing
    pd_bbox = None
    ocr_attempts = [
        # Original enhanced images
        {'desc': 'psm6', 'config': '--psm 6', 'img': enhanced},
        {'desc': 'psm7', 'config': '--psm 7', 'img': enhanced},
        {'desc': 'psm8', 'config': '--psm 8', 'img': enhanced},
        {'desc': 'psm10', 'config': '--psm 10 -c tessedit_char_whitelist=PD', 'img': enhanced},
        {'desc': 'psm11', 'config': '--psm 11', 'img': enhanced},
        # Inverted
        {'desc': 'psm6_inv', 'config': '--psm 6', 'img': 255 - enhanced},
        {'desc': 'psm7_inv', 'config': '--psm 7', 'img': 255 - enhanced},
        {'desc': 'psm10_inv', 'config': '--psm 10 -c tessedit_char_whitelist=PD', 'img': 255 - enhanced},
        # CLAHE enhanced
        {'desc': 'psm6_clahe', 'config': '--psm 6', 'img': enhanced_clahe},
        {'desc': 'psm7_clahe', 'config': '--psm 7', 'img': enhanced_clahe},
        {'desc': 'psm10_clahe', 'config': '--psm 10 -c tessedit_char_whitelist=PD', 'img': enhanced_clahe},
        # Simple threshold
        {'desc': 'psm6_otsu', 'config': '--psm 6', 'img': thresh_simple},
        {'desc': 'psm7_otsu', 'config': '--psm 7', 'img': thresh_simple},
        {'desc': 'psm10_otsu', 'config': '--psm 10 -c tessedit_char_whitelist=PD', 'img': thresh_simple},
        # Original gray (sometimes works better)
        {'desc': 'psm6_gray', 'config': '--psm 6', 'img': gray},
        {'desc': 'psm7_gray', 'config': '--psm 7', 'img': gray},
        {'desc': 'psm10_gray', 'config': '--psm 10 -c tessedit_char_whitelist=PD', 'img': gray},
    ]
    
    # Try pytesseract with STRICT matching - only "PD" or "P D"
    for attempt in ocr_attempts:
        ocr_result = pytesseract.image_to_data(attempt['img'], config=attempt['config'], output_type=pytesseract.Output.DICT)
        for i, text in enumerate(ocr_result['text']):
            text_original = text.strip().upper()
            text_clean = text_original.replace(' ', '')
            # STRICT matching: ONLY "PD" or "P D" (with single space)
            if text_clean == 'PD' or text_original == 'P D':
                x, y, w, h = ocr_result['left'][i], ocr_result['top'][i], ocr_result['width'][i], ocr_result['height'][i]
                # Validate bbox (should be reasonable size)
                if w > 5 and h > 5 and w < middle_crop.shape[1] * 0.3 and h < middle_crop.shape[0] * 0.5:
                    pd_bbox = (x, y, w, h)
                    # Visualize PD bbox
                    vis = cv2.cvtColor(attempt['img'], cv2.COLOR_GRAY2BGR) if len(attempt['img'].shape) == 2 else attempt['img'].copy()
                    if len(vis.shape) == 2:
                        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_4pd_bbox_{attempt["desc"]}.png'), vis)
                    break
        if pd_bbox is not None:
            break
    
    # Try EasyOCR as fallback - STRICT matching
    if pd_bbox is None:
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            # Try on different preprocessed images
            for img_variant, desc in [(enhanced, 'enhanced'), (gray, 'gray'), (enhanced_clahe, 'clahe')]:
                results = reader.readtext(img_variant)
                for (bbox, text, conf) in results:
                    text_original = text.strip().upper()
                    text_clean = text_original.replace(' ', '')
                    # STRICT: Only "PD" or "P D"
                    if text_clean == 'PD' or text_original == 'P D':
                        # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                        x_coords = [pt[0] for pt in bbox]
                        y_coords = [pt[1] for pt in bbox]
                        x = int(min(x_coords))
                        y = int(min(y_coords))
                        w = int(max(x_coords) - min(x_coords))
                        h = int(max(y_coords) - min(y_coords))
                        if w > 5 and h > 5 and w < middle_crop.shape[1] * 0.3 and h < middle_crop.shape[0] * 0.5:
                            pd_bbox = (x, y, w, h)
                            vis = cv2.cvtColor(img_variant, cv2.COLOR_GRAY2BGR) if len(img_variant.shape) == 2 else img_variant.copy()
                            if len(vis.shape) == 2:
                                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_4pd_bbox_easyocr_{desc}.png'), vis)
                            break
                if pd_bbox is not None:
                    break
        except Exception as e:
            print(f'EasyOCR fallback failed: {e}')
    
    # Final fallback: search in center region using pattern matching
    if pd_bbox is None:
        # PD is typically in the center horizontally
        center_x = middle_crop.shape[1] // 2
        center_region_w = int(middle_crop.shape[1] * 0.5)  # Wider search
        center_region = middle_crop[:, center_x - center_region_w//2:center_x + center_region_w//2]
        center_gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY) if len(center_region.shape) == 3 else center_region
        
        # Try multiple preprocessing on center region
        center_variants = [
            cv2.adaptiveThreshold(center_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10),
            cv2.adaptiveThreshold(center_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10),
            center_gray,
        ]
        
        for variant_idx, center_img in enumerate(center_variants):
            # Try multiple OCR configs on center region
            center_configs = [
                '--psm 10 -c tessedit_char_whitelist=PD',
                '--psm 8 -c tessedit_char_whitelist=PD',
                '--psm 7',
                '--psm 6',
            ]
            
            for config in center_configs:
                try:
                    ocr_result = pytesseract.image_to_data(center_img, config=config, output_type=pytesseract.Output.DICT)
                    for i, text in enumerate(ocr_result['text']):
                        text_original = text.strip().upper()
                        text_clean = text_original.replace(' ', '')
                        # STRICT: Only "PD" or "P D"
                        if text_clean == 'PD' or text_original == 'P D':
                            x_rel, y, w, h = ocr_result['left'][i], ocr_result['top'][i], ocr_result['width'][i], ocr_result['height'][i]
                            x = center_x - center_region_w//2 + x_rel
                            if w > 5 and h > 5 and w < middle_crop.shape[1] * 0.3:
                                pd_bbox = (x, y, w, h)
                                vis = middle_crop.copy()
                                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_4pd_bbox_center_fallback_v{variant_idx}.png'), vis)
                                break
                    if pd_bbox is not None:
                        break
                except:
                    continue
            if pd_bbox is not None:
                break
        
        # Try EasyOCR on center region as last resort
        if pd_bbox is None:
            try:
                reader = easyocr.Reader(['en'], gpu=False)
                for center_img in [center_gray, center_variants[0]]:
                    results = reader.readtext(center_img)
                    for (bbox, text, conf) in results:
                        text_original = text.strip().upper()
                        text_clean = text_original.replace(' ', '')
                        # STRICT: Only "PD" or "P D"
                        if text_clean == 'PD' or text_original == 'P D':
                            x_coords = [pt[0] for pt in bbox]
                            y_coords = [pt[1] for pt in bbox]
                            x_rel = int(min(x_coords))
                            y = int(min(y_coords))
                            w = int(max(x_coords) - min(x_coords))
                            h = int(max(y_coords) - min(y_coords))
                            x = center_x - center_region_w//2 + x_rel
                            if w > 5 and h > 5 and w < middle_crop.shape[1] * 0.3:
                                pd_bbox = (x, y, w, h)
                                vis = middle_crop.copy()
                                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_4pd_bbox_center_easyocr.png'), vis)
                                break
                    if pd_bbox is not None:
                        break
            except Exception as e:
                print(f'Center EasyOCR fallback failed: {e}')
    
    if pd_bbox is None:
        # Save all OCR text for debugging
        with open(os.path.join(output_dir, f'{prefix}_{now}_ocr_texts.txt'), 'w') as f:
            for attempt in ocr_attempts[:100]:  # Save first 100 attempts
                try:
                    ocr_result = pytesseract.image_to_string(attempt['img'], config=attempt['config'])
                    f.write(f"Attempt {attempt['desc']}\n{ocr_result}\n\n")
                except:
                    pass
        raise Exception('PD label not found by OCR.')

    # Step 4: Use edge detection to find the PD cell boundaries
    x, y, w, h = pd_bbox
    # Limit expansion: only slightly beyond detected PD text, avoid expanding downward
    expand_x = int(w ^ 2)
    expand_y = int(h ^ 1)
    search_x1 = max(0, x - expand_x)
    search_x2 = min(middle_crop.shape[1], x + w + expand_x)
    search_y1 = max(0, y - expand_y)
    search_y2 = min(middle_crop.shape[0], y + expand_y)  # Only a little below PD text

    # Extract region around PD
    pd_region = middle_crop[search_y1:search_y2, search_x1:search_x2]
    pd_region_gray = cv2.cvtColor(pd_region, cv2.COLOR_BGR2GRAY) if len(pd_region.shape) == 3 else pd_region

    # Use edge detection to find cell boundaries
    edges = cv2.Canny(pd_region_gray, 50, 150, apertureSize=3)
    # Use morphological operations to connect edges
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours to detect rectangular cell
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour that contains the PD text
    pd_cell_bbox = None
    pd_text_center_x = x - search_x1 + w // 2
    pd_text_center_y = y - search_y1 + h // 2

    for cnt in contours:
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
        # Check if PD text center is inside this contour
        if (x_cnt <= pd_text_center_x <= x_cnt + w_cnt and 
            y_cnt <= pd_text_center_y <= y_cnt + h_cnt):
            # This is likely the PD cell - use it
            pd_cell_bbox = (x_cnt, y_cnt, w_cnt, h_cnt)
            break

    # If no containing contour found, use the largest rectangular contour near PD
    if pd_cell_bbox is None:
        max_area = 0
        for cnt in contours:
            x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
            area = w_cnt * h_cnt
            # Check if it's near the PD text
            center_x = x_cnt + w_cnt // 2
            center_y = y_cnt + h_cnt // 2
            dist = np.sqrt((center_x - pd_text_center_x)**2 + (center_y - pd_text_center_y)**2)
            if dist < max(w, h) * 2 and area > max_area:
                max_area = area
                pd_cell_bbox = (x_cnt, y_cnt, w_cnt, h_cnt)

    # If still no cell found, use a tight bbox around PD text
    if pd_cell_bbox is None:
        cell_margin = max(w, h) * 0.2
        pd_cell_bbox = (
            max(0, (x - search_x1) - int(cell_margin)),
            max(0, (y - search_y1) - int(cell_margin)),
            w + int(cell_margin * 2),
            h + int(cell_margin * 2)
        )

    # Convert PD cell bbox to middle_crop coordinates
    pd_cell_x1 = search_x1 + pd_cell_bbox[0]
    pd_cell_y1 = search_y1 + pd_cell_bbox[1]
    pd_cell_x2 = pd_cell_x1 + pd_cell_bbox[2]
    pd_cell_y2 = pd_cell_y1 + pd_cell_bbox[3]
    
    # Visualize PD cell
    vis_cell = middle_crop.copy()
    cv2.rectangle(vis_cell, (pd_cell_x1, pd_cell_y1), (pd_cell_x2, pd_cell_y2), (0, 255, 0), 2)
    cv2.putText(vis_cell, 'PD Cell', (pd_cell_x1, pd_cell_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_4pd_cell_detected.png'), vis_cell)
    
    # Step 5: Find value cell below PD cell using edge detection
    value_search_y1 = pd_cell_y2 + 2
    value_search_y2 = min(middle_crop.shape[0], value_search_y1 + int(pd_cell_bbox[3] * 2.5))
    value_search_x1 = max(0, pd_cell_x1 - 5)
    value_search_x2 = min(middle_crop.shape[1], pd_cell_x2 + 5)
    
    value_search_region = middle_crop[value_search_y1:value_search_y2, value_search_x1:value_search_x2]
    value_region_gray = cv2.cvtColor(value_search_region, cv2.COLOR_BGR2GRAY) if len(value_search_region.shape) == 3 else value_search_region
    
    # Edge detection for value cell
    value_edges = cv2.Canny(value_region_gray, 50, 150, apertureSize=3)
    value_edges_dilated = cv2.dilate(value_edges, kernel, iterations=2)
    
    # Find contours for value cell
    value_contours, _ = cv2.findContours(value_edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    value_bbox = None
    max_area = 0
    
    for cnt in value_contours:
        x2, y2, w2, h2 = cv2.boundingRect(cnt)
        area = w2 * h2
        # Heuristic: look for rectangular box with reasonable aspect ratio
        aspect = w2 / h2 if h2 > 0 else 0
        if 1.0 < aspect < 5.0 and area > max_area and w2 > 10 and h2 > 5:
            max_area = area
            value_bbox = (x2, y2, w2, h2)
    
    if value_bbox is not None:
        # Adjust bbox to middle_crop coordinates
        value_x1 = value_search_x1 + value_bbox[0]
        value_y1 = value_search_y1 + value_bbox[1]
        value_x2 = value_x1 + value_bbox[2]
        value_y2 = value_y1 + value_bbox[3]
    else:
        # Fallback: use area below PD cell
        value_x1 = pd_cell_x1
        value_x2 = pd_cell_x2
        value_y1 = pd_cell_y2 + 2
        value_y2 = value_y1 + int(pd_cell_bbox[3] * 1.5)
    
    value_crop = gray[value_y1:value_y2, value_x1:value_x2]
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_5value_crop.png'), value_crop)
    
    # Mark both PD cell and value box on middle_crop visualization
    vis_middle = middle_crop.copy()
    # Mark PD cell in green
    cv2.rectangle(vis_middle, (pd_cell_x1, pd_cell_y1), (pd_cell_x2, pd_cell_y2), (0, 255, 0), 2)
    cv2.putText(vis_middle, 'PD Cell', (pd_cell_x1, pd_cell_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # Mark value box in red
    cv2.rectangle(vis_middle, (value_x1, value_y1), (value_x2, value_y2), (0, 0, 255), 2)
    cv2.putText(vis_middle, 'Value', (value_x1, value_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_5pd_and_value_bbox.png'), vis_middle)
    
    # Also mark bounding boxes on the original ROI_0 image (with respect to ROI_0 coordinates)
    # PD cell coordinates relative to ROI_0
    roi0_pd_x1 = pd_cell_x1
    roi0_pd_y1 = y1 + pd_cell_y1
    roi0_pd_x2 = pd_cell_x2
    roi0_pd_y2 = y1 + pd_cell_y2
    
    # Value box coordinates relative to ROI_0
    roi0_value_x1 = value_x1
    roi0_value_y1 = y1 + value_y1
    roi0_value_x2 = value_x2
    roi0_value_y2 = y1 + value_y2
    
    # Draw both on ROI_0 image
    roi0_img_copy = img.copy()
    # PD cell in green
    cv2.rectangle(roi0_img_copy, (roi0_pd_x1, roi0_pd_y1), (roi0_pd_x2, roi0_pd_y2), (0, 255, 0), 2)
    cv2.putText(roi0_img_copy, 'PD', (roi0_pd_x1, roi0_pd_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # Value box in red
    cv2.rectangle(roi0_img_copy, (roi0_value_x1, roi0_value_y1), (roi0_value_x2, roi0_value_y2), (0, 0, 255), 2)
    cv2.putText(roi0_img_copy, 'Value', (roi0_value_x1, roi0_value_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{now}_6pd_and_value_on_roi0.png'), roi0_img_copy)
    
    # Save value box coordinates in txt file (x, y, w, h) with respect to ROI_0 (overwrite PD cell bbox logic)
    roi0_value_bbox = [roi0_value_x1, roi0_value_y1, roi0_value_x2 - roi0_value_x1, roi0_value_y2 - roi0_value_y1]
    with open(os.path.join(output_dir, f'{prefix}_{now}_6value_bbox.txt'), 'w') as f:
        f.write(f'{roi0_value_bbox[0]},{roi0_value_bbox[1]},{roi0_value_bbox[2]},{roi0_value_bbox[3]}\n')

    # Step 6: OCR for value (digits and dot only) using EasyOCR
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

    # Step 7: Output
    result = {
        'pd_value_bbox': [int(roi0_value_bbox[0]), int(roi0_value_bbox[1]), int(roi0_value_bbox[2]), int(roi0_value_bbox[3])],  # Value box bbox on ROI_0
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
