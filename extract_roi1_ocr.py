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

# If run as a script, keep original behavior
if __name__ == '__main__':
    import sys
    import json
    import re
    # Usage: python extract_roi1_ocr.py [roi1_img_path] [bbox_path]
    if len(sys.argv) >= 3:
        roi1_path = sys.argv[1]
        bbox_path = sys.argv[2]
        img = cv2.imread(roi1_path)
        if img is None:
            raise FileNotFoundError(f'Could not load {roi1_path}')
        def parse_bbox_line(line):
            nums = re.findall(r'(?:np\.int64\()?(-?\d+)(?:\))?', line)
            return tuple(map(int, nums))
        with open(bbox_path, 'r') as f:
            bboxes = [parse_bbox_line(line) for line in f]
        results = extract_roi1_ocr(img, bboxes)
        roi1_dir = os.path.dirname(roi1_path) or '.'
    else:
        roi1_dir = 'ROI_1'
        roi1_files = [f for f in os.listdir(roi1_dir) if f.startswith('roi1_') and f.endswith('.png')]
        roi1_files.sort()
        roi1_path = os.path.join(roi1_dir, roi1_files[-1])
        bbox_files = [f for f in os.listdir(roi1_dir) if f.startswith('roi1_bboxes_') and f.endswith('.txt')]
        bbox_files.sort()
        bbox_path = os.path.join(roi1_dir, bbox_files[-1])
        img = cv2.imread(roi1_path)
        if img is None:
            raise FileNotFoundError(f'Could not load {roi1_path}')
        def parse_bbox_line(line):
            nums = re.findall(r'(?:np\.int64\()?(-?\d+)(?:\))?', line)
            return tuple(map(int, nums))
        with open(bbox_path, 'r') as f:
            bboxes = [parse_bbox_line(line) for line in f]
        results = extract_roi1_ocr(img, bboxes)
    for label in [
        'R_Sph', 'S_Anchor', 'L_Sph',
        'R_Cyl', 'C_Anchor', 'L_Cyl',
        'R_Axis', 'A_Anchor', 'L_Axis',
        'R_Add', 'ADD_Anchor', 'L_Add'
    ]:
        print(f'{label}: {results[label]}')
    now = datetime.datetime.now().strftime('%d%m_%H%M%S')
    # Get prefix from roi1_path
    base = os.path.splitext(os.path.basename(roi1_path))[0]
    prefix = base[:4]
    output_path = os.path.join(roi1_dir, f'{prefix}_{now}_ocr.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Cell values saved to {output_path}')


