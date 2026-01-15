import cv2
import numpy as np
import os

# Load latest ROI-0 and ROI-1 images
roi0_dir = 'ROI_0'
roi1_dir = 'ROI_1'
roi0_files = [f for f in os.listdir(roi0_dir) if f.startswith('roi0_') and f.endswith('.png')]
roi1_files = [f for f in os.listdir(roi1_dir) if f.startswith('roi1_') and f.endswith('.png')]
roi0_files.sort()
roi1_files.sort()
roi0_path = os.path.join(roi0_dir, roi0_files[-1])
roi1_path = os.path.join(roi1_dir, roi1_files[-1])

roi0 = cv2.imread(roi0_path)
roi1 = cv2.imread(roi1_path)

# Load the bounding box of ROI-1 in ROI-0 coordinates
# Use the top half crop and table detection logic from previous steps
# Find the location of ROI-1 inside the top half of ROI-0
# (Assume the same detection logic as in extract_roi1_ocr.py)

# Load the top half crop used for ROI-1 detection
roi0_top_half_files = [f for f in os.listdir(roi1_dir) if f.startswith('roi0_top_half_') and f.endswith('.png')]
roi0_top_half_files.sort()
roi0_top_half_path = os.path.join(roi1_dir, roi0_top_half_files[-1])
roi0_top_half = cv2.imread(roi0_top_half_path)

# Find where ROI-1 is located in the top half crop
result = cv2.matchTemplate(roi0_top_half, roi1, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
roi1_in_top_half_x, roi1_in_top_half_y = max_loc

# Now, find where the top half crop is located in ROI-0 (should be at y=0, but check)
result2 = cv2.matchTemplate(roi0, roi0_top_half, cv2.TM_CCOEFF_NORMED)
_, _, _, max_loc2 = cv2.minMaxLoc(result2)
top_half_in_roi0_x, top_half_in_roi0_y = max_loc2

# Calculate the absolute position of ROI-1 in ROI-0
roi1_abs_x = top_half_in_roi0_x + roi1_in_top_half_x
roi1_abs_y = top_half_in_roi0_y + roi1_in_top_half_y

# Overlay ROI-1 grid lines on ROI-0
overlay = roi0.copy()
# Load grid bounding boxes
bbox_files = [f for f in os.listdir(roi1_dir) if f.startswith('roi1_bboxes_') and f.endswith('.txt')]
bbox_files.sort()
bbox_path = os.path.join(roi1_dir, bbox_files[-1])
with open(bbox_path, 'r') as f:
    bboxes = [tuple(map(int, line.strip('()\n').split(','))) for line in f]

# Draw each grid line/cell on the overlay at the correct absolute position
for bbox in bboxes:
    x1, y1, x2, y2 = bbox
    abs_x1 = roi1_abs_x + x1
    abs_y1 = roi1_abs_y + y1
    abs_x2 = roi1_abs_x + x2
    abs_y2 = roi1_abs_y + y2
    cv2.rectangle(overlay, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)

# Optionally, draw the outer ROI-1 rectangle
h1, w1 = roi1.shape[:2]
cv2.rectangle(overlay, (roi1_abs_x, roi1_abs_y), (roi1_abs_x + w1, roi1_abs_y + h1), (0, 255, 0), 2)

# Save the overlay image
output_path = os.path.join(roi0_dir, 'roi1_overlay_on_roi0.png')
cv2.imwrite(output_path, overlay)
print(f'ROI-1 grid overlay saved to {output_path}')
