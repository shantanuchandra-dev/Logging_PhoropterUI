"""
This script extracts the menu panel (top bar) from the latest ROI-0 image and saves it as ROI_Menu.
"""
import cv2
import numpy as np
import pytesseract
import os
import datetime

# Path to the latest ROI-0 image (assume most recent file in ROI_0)
roi0_dir = 'ROI_0'
roi0_files = [f for f in os.listdir(roi0_dir) if f.startswith('roi0_') and f.endswith('.png') and 'box' not in f]
if not roi0_files:
    raise FileNotFoundError('No ROI-0 images found in ROI_0 directory.')
roi0_files.sort()
roi0_path = os.path.join(roi0_dir, roi0_files[-1])

img = cv2.imread(roi0_path)
if img is None:
    raise FileNotFoundError(f'Could not load {roi0_path}')

# Convert to grayscale and blur
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest rectangular contour near the top (Menu panel)
menu_contour = None
max_area = 0
img_h, img_w = img.shape[:2]
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    # Heuristic: Menu is at the very top, wide, and rectangular
    if len(approx) == 4 and area > max_area and y < img_h // 8 and w > img_w // 2:
        menu_contour = approx
        max_area = area

if menu_contour is None:
    raise Exception('Menu panel not found.')

# Get bounding rect and crop
x, y, w, h = cv2.boundingRect(menu_contour)
menu = img[y:y+h, x:x+w]

# Save Menu crop
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = 'ROI_Menu'
os.makedirs(output_dir, exist_ok=True)
menu_path = os.path.join(output_dir, f'roi_menu_{now}.png')
cv2.imwrite(menu_path, menu)
print(f'Menu panel saved to {menu_path}')

# OCR the Menu panel (optional)
ocr_config = '--psm 6'
txt = pytesseract.image_to_string(menu, config=ocr_config)
ocr_txt_path = os.path.join(output_dir, f'roi_menu_ocr_{now}.txt')
with open(ocr_txt_path, 'w') as f:
    f.write(txt)
print(f'OCR result saved to {ocr_txt_path}')
