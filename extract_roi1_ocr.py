import cv2
import numpy as np
import pytesseract
import os
import datetime
import re

def extract_roi1_ocr(img, bboxes):
    """
    Extracts OCR values from ROI-1 table cells using absolute bboxes on ROI0.
    img: The full ROI0 image (not a cropped table image!)
    bboxes: List of (x1, y1, x2, y2) absolute coordinates on ROI0.
    """
    results = {}
    cell_labels = [
        ['R_Sph', 'S_Anchor', 'L_Sph'],
        ['R_Cyl', 'C_Anchor', 'L_Cyl'],
        ['R_Axis', 'A_Anchor', 'L_Axis'],
        ['R_Add', 'ADD_Anchor', 'L_Add'],
        ['R_blank', 'blank_anchor', 'L_blank']
    ]
    
    # Filter bboxes if we have more than 15 cells
    n_cols = 3
    n_expected = 15
    if len(bboxes) > n_expected:
        # Group by rows and find starting row
        n_rows_detected = len(bboxes) // n_cols
        rows = []
        for i in range(n_rows_detected):
            row_bboxes = bboxes[i * n_cols:(i + 1) * n_cols]
            rows.append(row_bboxes)
        
        # Find starting row where 2nd cell has 'S'
        start_row_idx = None
        for i, row_bboxes in enumerate(rows):
            if len(row_bboxes) >= 2:
                x1, y1, x2, y2 = row_bboxes[1]
                cell_img = img[y1:y2, x1:x2]
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                _, bin_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                text = pytesseract.image_to_string(bin_img, config='--oem 3 --psm 10 -c tessedit_char_whitelist=S')
                text = text.strip().upper()
                if 'S' in text:
                    start_row_idx = i
                    break
        
        # If no 'S' found, skip rows with 'R' in 1st or 'L' in 3rd cell
        if start_row_idx is None:
            for i, row_bboxes in enumerate(rows):
                if len(row_bboxes) >= 3:
                    # Check 1st cell for 'R'
                    x1, y1, x2, y2 = row_bboxes[0]
                    cell_img = img[y1:y2, x1:x2]
                    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                    _, bin_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                    text1 = pytesseract.image_to_string(bin_img, config='--oem 3 --psm 10 -c tessedit_char_whitelist=R')
                    text1 = text1.strip().upper()
                    
                    # Check 3rd cell for 'L'
                    x1, y1, x2, y2 = row_bboxes[2]
                    cell_img = img[y1:y2, x1:x2]
                    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                    _, bin_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                    text3 = pytesseract.image_to_string(bin_img, config='--oem 3 --psm 10 -c tessedit_char_whitelist=L')
                    text3 = text3.strip().upper()
                    
                    if 'R' in text1 or 'L' in text3:
                        continue
                    else:
                        start_row_idx = i
                        break
        
        if start_row_idx is None:
            start_row_idx = 0
        
        # Take exactly 5 rows
        filtered_rows = rows[start_row_idx:start_row_idx + 5]
        bboxes = []
        for row_bboxes in filtered_rows:
            if len(row_bboxes) == n_cols:
                bboxes.extend(row_bboxes)
        
        # ...existing code...
    
    for row in range(5):  # Process all 5 rows including blank
        for col in range(3):
            idx = row * 3 + col
            if idx >= len(bboxes):
                results[cell_labels[row][col]] = None
                continue
            x1, y1, x2, y2 = bboxes[idx]
            cell_img = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            label = cell_labels[row][col]
            value = None
            if 'Sph' in label or 'Cyl' in label or 'Add' in label:
                configs = [
                    '--oem 3 --psm 10',
                    '--oem 3 --psm 8',
                    '--oem 3 --psm 7',
                ]
                for config in configs:
                    text = pytesseract.image_to_string(enhanced, config=config)
                    text = text.strip().replace(' ', '').replace('\n', '')
                    match = re.search(r'([+-]?\d+\.\d{2})', text)
                    if match:
                        value = match.group(1)
                        if value == '0.00' or value.startswith(('+', '-')):
                            break
                        else:
                            value = None
            elif 'Axis' in label:
                configs = [
                    '--oem 3 --psm 10',
                    '--oem 3 --psm 8',
                    '--oem 3 --psm 7',
                ]
                for config in configs:
                    text = pytesseract.image_to_string(enhanced, config=config)
                    text = text.strip().replace(' ', '').replace('\n', '')
                    match = re.search(r'(\d{2,3})', text)
                    if match:
                        val = int(match.group(1))
                        if val % 5 == 0:
                            value = str(val)
                            break
            elif 'Anchor' in label:
                _, bin_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                anchor_map = {
                    'S_Anchor': 'S',
                    'C_Anchor': 'C',
                    'A_Anchor': 'A',
                    'ADD_Anchor': 'ADD'
                }
                whitelist = anchor_map.get(label, '')
                if whitelist:
                    config = f'--oem 3 --psm 10 -c tessedit_char_whitelist={whitelist}'
                    text = pytesseract.image_to_string(bin_img, config=config)
                    value = text.strip().replace(' ', '').replace('\n', '')
                    if label == 'ADD_Anchor' and value.lower() != 'add':
                        value = None
                else:
                    value = ''
            else:
                custom_config = r'--oem 3 --psm 7'
                text = pytesseract.image_to_string(enhanced, config=custom_config)
                value = text.strip().replace(' ', '').replace('\n', '')
            results[label] = value

    # Print main OCR values in order after extraction
    main_keys = ['R_Sph', 'L_Sph', 'R_Cyl', 'L_Cyl', 'R_Axis', 'L_Axis', 'R_Add', 'L_Add']
    print(' | '.join(str(results.get(k, '')) for k in main_keys))

    # After extraction, check if majority of R/L Sph, Cyl, Axis, Add are blank/null
    main_keys = ['R_Sph', 'L_Sph', 'R_Cyl', 'L_Cyl', 'R_Axis', 'L_Axis', 'R_Add', 'L_Add']
    blank_count = sum(1 for k in main_keys if not results.get(k))
    if blank_count >= 5:
        def parse_bbox(b):
            if isinstance(b, (list, tuple)) and len(b) == 4:
                return tuple(int(x) for x in b)
            if isinstance(b, str):
                import re
                nums = re.findall(r'-?\d+', b)
                if len(nums) == 4:
                    return tuple(int(x) for x in nums)
            raise ValueError(f"Invalid bbox format: {b}")
        try:
            os.makedirs('ROI_1', exist_ok=True)
            idx_r_sph = 0  # row 0, col 0
            idx_l_sph = 2  # row 0, col 2
            if idx_r_sph < len(bboxes):
                x1, y1, x2, y2 = parse_bbox(bboxes[idx_r_sph])
                r_sph_img = img[y1:y2, x1:x2]
                cv2.imwrite('ROI_1/R_Sph_crop_cells_on_roi0.png', r_sph_img)
            if idx_l_sph < len(bboxes):
                x1, y1, x2, y2 = parse_bbox(bboxes[idx_l_sph])
                l_sph_img = img[y1:y2, x1:x2]
                cv2.imwrite('ROI_1/L_Sph_crop_cells_on_roi0.png', l_sph_img)
        except Exception as crop_exc:
            print(f"[ERROR] Could not save R_Sph/L_Sph crops: {crop_exc}")
        import sys
        print(f"[ERROR] extract_roi1_ocr: {blank_count} of 8 main Sph/Cyl/Axis/Add fields are blank/null. Likely bbox/image mismatch or OCR failure. Exiting.")
        sys.exit(1)
    return results


# Keep original main for testing
if __name__ == '__main__':
    import sys
    import json
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is not None:
            # If it's a full ROI-0 image
            if img.shape[0] > 600:
                res = extract(img, save_debug=True)
                # Removed debug print
            else:
                # If it's just the table crop
                if len(sys.argv) > 2:
                    bbox_path = sys.argv[2]
                    def parse_bbox_line(line):
                        nums = re.findall(r'(?:np\.int64\()?(-?\d+)(?:\))?', line)
                        return tuple(map(int, nums))
                    with open(bbox_path, 'r') as f:
                        bboxes = [parse_bbox_line(line) for line in f]
                    res = extract_roi1_ocr(img, bboxes)
                    # Removed debug print


