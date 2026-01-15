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

# --- Find ROI_Menu to crop it out completely and track offset ---
roi_menu_dir = 'ROI_MENU'
roi_menu_files = [f for f in os.listdir(roi_menu_dir) if f.startswith('roi_menu_') and f.endswith('.png')]
roi_menu_files.sort()
roi_menu_path = os.path.join(roi_menu_dir, roi_menu_files[-1]) if roi_menu_files else None
roi_menu_y2 = 0
if roi_menu_path:
    menu_img = cv2.imread(roi_menu_path)
    if menu_img is not None:
        roi_menu_y2 = menu_img.shape[0]
        # Crop out the menu from the image for processing, but keep offset for output
        img_cropped = img[roi_menu_y2:, :]
    else:
        print('Warning: Could not load ROI_Menu image, not cropping menu.')
        img_cropped = img
else:
    print('Warning: No ROI_Menu image found, not cropping menu.')
    img_cropped = img



# Crop ROI-0 to top half (minus ROI_Menu)
h_full, w_full = img_cropped.shape[:2]
roi0_top_half = img_cropped[:h_full//2, :]

# Save the cropped top half image
output_dir = 'ROI_1'
os.makedirs(output_dir, exist_ok=True)
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
top_half_path = os.path.join(output_dir, f'roi0_top_half_{now}.png')
cv2.imwrite(top_half_path, roi0_top_half)
print(f'Cropped ROI-0 top half saved to {top_half_path}')


# --- Improved 3x5 Table Detection ---
gray = cv2.cvtColor(roi0_top_half, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)

# Morphological operations to find horizontal and vertical lines
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h, iterations=2)
vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v, iterations=2)

# Combine lines to get grid intersections
grid = cv2.add(horizontal, vertical)
contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest rectangle that could be a table
max_area = 0
table_rect = None
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    aspect = w / float(h)
    # Heuristic: Table is wide, not too thin, and large
    if area > max_area and 1.5 < aspect < 3.5 and w > 100 and h > 50:
        max_area = area
        table_rect = (x, y, w, h)

if table_rect is None:
    raise Exception('No 3x5 table-like rectangle found in ROI-0 top half.')

# Crop the detected table
x, y, w, h = table_rect
roi1 = roi0_top_half[y:y+h, x:x+w]
roi1_path = os.path.join(output_dir, f'roi1_{now}.png')
cv2.imwrite(roi1_path, roi1)
print(f'ROI-1 (table) saved to {roi1_path}')

# Save visualization with grid overlay
vis = roi0_top_half.copy()
cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
contour_vis_path = os.path.join(output_dir, f'roi1_grid_{now}.png')
cv2.imwrite(contour_vis_path, vis)
print(f'ROI-1 grid visualization saved to {contour_vis_path}')
