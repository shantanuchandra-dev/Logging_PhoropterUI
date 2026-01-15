import cv2
import numpy as np
import pytesseract
import os
import datetime
import re

# Load latest ROI-1 image and bounding boxes
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

with open(bbox_path, 'r') as f:
    bboxes = [tuple(map(int, line.strip('()\n').split(','))) for line in f]

# Cell labels
cell_labels = [
    ['R_Sph', 'S_Anchor', 'L_Sph'],
    ['R_Cyl', 'C_Anchor', 'L_Cyl'],
    ['R_Axis', 'A_Anchor', 'L_Axis'],
    ['R_Add', 'ADD_Anchor', 'L_Add'],
    ['R_blank', 'blank_anchor', 'L_blank']
]



# Only process first 4 rows (ignore blank row)
results = {}
for row in range(4):
    for col in range(3):
        idx = row * 3 + col
        x1, y1, x2, y2 = bboxes[idx]
        cell_img = img[y1:y2, x1:x2]
        # Enhance contrast
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        label = cell_labels[row][col]
        value = None
        conf = None
        # Sph, Cyl, Add: +/-0.25 increments, 0.00 can be without symbol
        if 'Sph' in label or 'Cyl' in label or 'Add' in label:
            # Try best configs for numbers
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
        # Axis: increments of 5, no symbol
        elif 'Axis' in label:
            # Try best configs for axis
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
        # Anchor cells: use binarization and whitelist
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

# Print results
for label in [
    'R_Sph', 'S_Anchor', 'L_Sph',
    'R_Cyl', 'C_Anchor', 'L_Cyl',
    'R_Axis', 'A_Anchor', 'L_Axis',
    'R_Add', 'ADD_Anchor', 'L_Add'
]:
    print(f'{label}: {results[label]}')

# Save results
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = os.path.join(roi1_dir, f'ROI1_OCR_{now}.json')
import json
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Cell values saved to {output_path}')
