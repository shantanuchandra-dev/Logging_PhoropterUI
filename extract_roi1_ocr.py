import cv2
import numpy as np
import pytesseract
import os
import datetime
import re

def extract_roi1_ocr(img, bboxes):
    import pytesseract
    import re
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
        
        print(f'Filtered bboxes for OCR: {len(bboxes)} cells (started from row {start_row_idx})')
    
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
    return results

def extract(roi0_img, save_debug=False, output_dir='ROI_1', filename=None):
    """
    Standard extract function for pipeline integration.
    Finds the S/C/A/ADD table in ROI-0 and performs OCR.
    """
    h_img, w_img = roi0_img.shape[:2]
    
    # 1. Find the Table ROI (ROI-1)
    # The table is typically in the upper-center area.
    search_y_end = int(h_img * 0.5)
    search_x_start = int(w_img * 0.1)
    search_x_end = int(w_img * 0.9)
    
    crop = roi0_img[0:search_y_end, search_x_start:search_x_end]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    table_rect = None
    max_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Table must be large and roughly rectangular/central
        if area > 10000 and 0.5 < w/h < 2.5:
            if area > max_area:
                max_area = area
                table_rect = (x + search_x_start, y, w, h)
                
    if not table_rect:
        # Fallback: use a plausible central region if detection fails
        table_rect = (int(w_img * 0.25), int(h_img * 0.1), int(w_img * 0.5), int(h_img * 0.35))

    tx, ty, tw, th = table_rect
    table_img = roi0_img[ty:ty+th, tx:tx+tw]
    
    # 2. Segment Cells (3 columns, 5 rows)
    # Using simple proportional segmentation as fallback, but let's try to find dividers
    gray_table = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
    edges_table = cv2.Canny(gray_table, 50, 150)
    
    proj_h = np.sum(edges_table, axis=1)
    proj_v = np.sum(edges_table, axis=0)
    
    def find_dividers(proj, n_segments, min_gap=10):
        if len(proj) == 0: return []
        limit = np.max(proj) * 0.2
        candidates = np.where(proj > limit)[0]
        if len(candidates) == 0: return []
        clusters = []
        if len(candidates) > 0:
            curr = [candidates[0]]
            for i in range(1, len(candidates)):
                if candidates[i] - candidates[i-1] < min_gap:
                    curr.append(candidates[i])
                else:
                    clusters.append(int(np.mean(curr)))
                    curr = [candidates[i]]
            clusters.append(int(np.mean(curr)))
        return clusters

    row_divs = find_dividers(proj_h, 5)
    col_divs = find_dividers(proj_v, 3)
    
    # Ensure we have enough dividers or use proportional
    if len(row_divs) < 4:
        row_divs = [int(th * i / 5) for i in range(6)]
    else:
        if row_divs[0] > 10: row_divs = [0] + row_divs
        if row_divs[-1] < th - 10: row_divs.append(th)
        
    if len(col_divs) < 2:
        col_divs = [int(tw * i / 3) for i in range(4)]
    else:
        if col_divs[0] > 10: col_divs = [0] + col_divs
        if col_divs[-1] < tw - 10: col_divs.append(tw)

    # Generate cell bboxes relative to table_img
    bboxes = []
    for r in range(len(row_divs) - 1):
        for c in range(len(col_divs) - 1):
            y1, y2 = row_divs[r], row_divs[r+1]
            x1, x2 = col_divs[c], col_divs[c+1]
            bboxes.append((x1, y1, x2, y2))
            
    # 3. Perform OCR
    ocr_results = extract_roi1_ocr(table_img, bboxes)
    
    result = {
        'roi_id': 'ROI_1',
        'bbox': [int(tx), int(ty), int(tw), int(th)],
        'data': ocr_results
    }
    
    if save_debug:
        os.makedirs(output_dir, exist_ok=True)
        now = datetime.datetime.now().strftime('%d%m_%H%M%S')
        if filename:
            input_base = os.path.splitext(os.path.basename(filename))[0]
            prefix = input_base[:4]
        else:
            prefix = 'roi1'
        viz = roi0_img.copy()
        cv2.rectangle(viz, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 2)
        for box in bboxes:
            bx1, by1, bx2, by2 = box
            cv2.rectangle(viz, (tx+bx1, ty+by1), (tx+bx2, ty+by2), (255, 0, 0), 1)
        viz_path = os.path.join(output_dir, f'{prefix}_{now}_viz_roi1.png')
        cv2.imwrite(viz_path, viz)
        result['image_path'] = viz_path
        
    return result

# Keep original main for testing
if __name__ == '__main__':
    # (Original main content remains similar but uses extract if needed)
    import sys
    import json
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is not None:
            # If it's a full ROI-0 image
            if img.shape[0] > 600:
                res = extract(img, save_debug=True)
                print(json.dumps(res, indent=2))
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
                    print(json.dumps(res, indent=2))


